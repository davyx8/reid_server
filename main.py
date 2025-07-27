import cv2
import numpy as np
import torch
import torch.nn as nn
import argparse
from collections import defaultdict

# Ultralytics YOLO for detection
from ultralytics import YOLO
# TorchReID for person re-identification
import torchreid

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Multi-camera person tracking with re-identification"
)
parser.add_argument("video1", help="Path to first input video")
parser.add_argument("video2", help="Path to second input video")
parser.add_argument("--output1", default="video1_annotated.mp4",
                    help="Output path for annotated first video")
parser.add_argument("--output2", default="video2_annotated.mp4",
                    help="Output path for annotated second video")
parser.add_argument("--csv", default="appearances.csv",
                    help="Output CSV file for appearances")
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Load detection model (YOLOv8)
# -----------------------------------------------------------------------------
detect_model = YOLO('yolov8n.pt')  # downloads automatically if needed

# -----------------------------------------------------------------------------
# Load Re-ID model (OSNet_x0_25) via torchreid
# -----------------------------------------------------------------------------
reid_model = torchreid.models.build_model(
    name='osnet_x0_25',
    num_classes=1000,
    loss='softmax',
    pretrained=True
)
# Strip classifier to output 512-D embeddings
reid_model.classifier = nn.Identity()
reid_model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
reid_model.to(device)


# -----------------------------------------------------------------------------
# Utilities: detection and Re-ID feature extraction
# -----------------------------------------------------------------------------
def detect_frame(frame):
    """
    Returns:
        person_boxes  – list of (x,y,w,h)
        obj_per_pbox  – list of sets with object class names for each person box
    """
    results = detect_model(frame[..., ::-1])  # BGR→RGB
    boxes = results[0].boxes
    pboxes, pobj_sets = [], []

    # Separate detections
    persons, objects = [], []
    for box, cls, conf in zip(boxes.xyxy.cpu().numpy(),
                              boxes.cls.cpu().numpy(),
                              boxes.conf.cpu().numpy()):
        if conf < 0.35:  # looser for objects
            continue
        (x1, y1, x2, y2) = map(int, box)
        if int(cls) == 0:
            persons.append(((x1, y1, x2, y2), conf))
        else:
            objects.append(((x1, y1, x2, y2), int(cls)))

    # For each person, collect overlapped objects
    for (x1, y1, x2, y2), _ in persons:
        box_area = (x2 - x1) * (y2 - y1)
        oset = set()
        for (ox1, oy1, ox2, oy2), ocls in objects:
            ix1, iy1 = max(x1, ox1), max(y1, oy1)
            ix2, iy2 = min(x2, ox2), min(y2, oy2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter / float(box_area) >= 0.30:
                oset.add(results[0].names[ocls])  # class → name
        pboxes.append((x1, y1, x2 - x1, y2 - y1))
        pobj_sets.append(oset)

    return pboxes, pobj_sets


def extract_reid_feature(image):
    """
    Input: BGR crop (HxWx3).
    Output: L2-normalized 512-D feature vector.
    """
    img = cv2.resize(image, (128, 256))
    img = img[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = reid_model(tensor).squeeze(0).cpu().numpy()
    # L2 normalize
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm
    return feat


# -----------------------------------------------------------------------------
# Track class, now with video_id
# -----------------------------------------------------------------------------
next_track_id = 1


class Track:
    def __init__(self, bbox, feature, frame_idx, video_id, obj_set):
        global next_track_id
        self.local_id = next_track_id
        next_track_id += 1
        self.bbox = bbox
        self.feature_sum = feature.copy()
        self.feature_count = 1
        self.start_frame = frame_idx
        self.end_frame = frame_idx
        self.missed = 0
        self.active = True
        self.video_id = video_id  # either 1 or 2
        self.id = None  # final global ID

        self.base_feat_sum = feature.copy()  # first N features for baseline
        self.base_feat_count = 1
        self.obj_set = set(obj_set)  # objects seen so far
        self.anomalies = []  # list of (frame_idx, type, details)
        self.clothes_changed = False

    def mark_missed(self):
        self.missed += 1
        if self.missed > 30:
            self.active = False

    def avg_feature(self):
        feat = self.feature_sum / self.feature_count
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        return feat

    def update(self, bbox, feature, frame_idx, obj_set):
        self.bbox = bbox
        self.end_frame = frame_idx
        self.missed = 0
        self.feature_sum += feature
        self.feature_count += 1
        # ---------- Apparel‑change check ----------
        if not self.clothes_changed:
            base_feat = self.base_feat_sum / self.base_feat_count
            sim = float(np.dot(base_feat, feature))
            if sim < 0.60:  # ● tweak threshold if needed
                print(f'Clothes change detected for Track {self.id} at frame {frame_idx}')
                self.anomalies.append((frame_idx, "clothes", f"cos={sim:.2f}"))
                self.clothes_changed = True
        else:
            # keep baseline stable after first change
            pass
        # accumulate baseline while it is still "clean"
        if not self.clothes_changed and self.base_feat_count < 20:
            self.base_feat_sum += feature
            self.base_feat_count += 1

        # ---------- Object‑set change check ----------
        new_objs = set(obj_set)
        if new_objs != self.obj_set:
            diff = new_objs.symmetric_difference(self.obj_set)
            print(f'Object set change detected for Track {self.id} at frame {frame_idx}: {diff}')
            self.anomalies.append((frame_idx, "object", f"Δ={diff}"))
            self.obj_set = new_objs


# Containers
tracks_video1 = []
tracks_video2 = []
track_bboxes = defaultdict(list)  # Track -> list of (frame_idx, bbox)


# -----------------------------------------------------------------------------
# Process a single video: detect, track, record bboxes
# -----------------------------------------------------------------------------
def process_video(path, tracks, fps, video_id):
    cap = cv2.VideoCapture(path)
    frame_idx = 0
    global next_track_id

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        pboxes, pobj_sets = detect_frame(frame)
        updated = []

        for (bbox, obj_set) in zip(pboxes, pobj_sets):
            x, y, w, h = bbox
            crop = frame[y:y + h, x:x + w]
            feat = extract_reid_feature(crop)

            best, best_iou = None, 0
            for t in tracks:
                if not t.active:
                    continue
                # IoU
                tx, ty, tw, th = t.bbox
                ix1 = max(tx, x);
                iy1 = max(ty, y)
                ix2 = min(tx + tw, x + w);
                iy2 = min(ty + th, y + h)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                if inter == 0:
                    continue
                iou = inter / float(tw * th + w * h - inter)
                if iou > 0.3:
                    sim = np.dot(t.avg_feature(), feat)
                    if sim > 0.3 and iou > best_iou:
                        best_iou, best = iou, t
            if best:
                best.update((x, y, w, h), feat, frame_idx, obj_set)
                track = best
            else:
                track = Track((x, y, w, h), feat, frame_idx, video_id, obj_set)
                tracks.append(track)

            updated.append(track)
            track_bboxes[track].append((frame_idx, track.bbox))

        # mark missed
        for t in tracks:
            if t not in updated:
                t.mark_missed()

    cap.release()


# -----------------------------------------------------------------------------
# Run tracking on both videos
# -----------------------------------------------------------------------------
# fps values for time calculations
cap1 = cv2.VideoCapture(args.video1)
fps1 = cap1.get(cv2.CAP_PROP_FPS)
cap1.release()
cap2 = cv2.VideoCapture(args.video2)
fps2 = cap2.get(cv2.CAP_PROP_FPS)
cap2.release()

process_video(args.video1, tracks_video1, fps1, video_id=1)
# reset local IDs for video2
next_track_id = 1
process_video(args.video2, tracks_video2, fps2, video_id=2)

# -----------------------------------------------------------------------------
# Global matching across all tracks via Re-ID features
# Enforce no two overlapping tracks in same video share an ID
# -----------------------------------------------------------------------------
global_id = 0
# store global features and track lists
global_feats = []  # list of (gid, feature)
global_id_to_tracks = defaultdict(list)  # gid -> list of Track

track_to_global = {}

# helper to match a track to a global ID
SIM_THRESHOLD = 0.8  # stricter similarity threshold


def match_track(track):
    global global_id
    feat = track.avg_feature()
    best_gid = None
    best_sim = SIM_THRESHOLD
    # try existing global IDs\
    for gid, gf in global_feats:
        sim = float(np.dot(gf, feat))
        if sim < best_sim:
            continue
        # check temporal exclusivity within same video
        conflict = False
        for other in global_id_to_tracks[gid]:
            if other.video_id == track.video_id:
                # if time intervals overlap
                if not (track.end_frame < other.start_frame or track.start_frame > other.end_frame):
                    conflict = True
                    break
        if conflict:
            continue
        # accept this match
        best_gid = gid
        best_sim = sim

    if best_gid is not None:
        # update global feature (running avg)
        for i, (gid, gf) in enumerate(global_feats):
            if gid == best_gid:
                nf = gf + feat
                nf /= np.linalg.norm(nf)
                global_feats[i] = (gid, nf)
                break
        return best_gid

    # no match -> new global ID
    global_id += 1
    global_feats.append((global_id, feat))
    return global_id


# assign globals and record
for t in tracks_video1 + tracks_video2:
    gid = match_track(t)
    t.id = gid
    track_to_global[t] = gid
    global_id_to_tracks[gid].append(t)

# -----------------------------------------------------------------------------
# Build appearance intervals and write CSV
# -----------------------------------------------------------------------------
import csv

appearances = defaultdict(lambda: {'video1': [], 'video2': []})

for t in tracks_video1:
    appearances[t.id]['video1'].append((t.start_frame / fps1, t.end_frame / fps1))
for t in tracks_video2:
    appearances[t.id]['video2'].append((t.start_frame / fps2, t.end_frame / fps2))


# merge contiguous intervals
def merge_ints(ints):
    if not ints:
        return []
    ints.sort(key=lambda x: x[0])
    merged = []
    s, e = ints[0]
    for ns, ne in ints[1:]:
        if ns <= e + 1 / fps1:
            e = max(e, ne)
        else:
            merged.append((s, e))
            s, e = ns, ne
    merged.append((s, e))
    return merged


with open("anomalies.csv", "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["GlobalID", "Video", "Timestamp(s)", "Type", "Details"])
    for t in tracks_video1 + tracks_video2:
        fps = fps1 if t.video_id == 1 else fps2
        for fr, typ, det in t.anomalies:
            wr.writerow([t.id, t.video_id, f"{fr / fps:.2f}", typ, det])

with open(args.csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['PersonID', 'Video1_Appearances', 'Video2_Appearances'])
    for gid, vids in appearances.items():
        v1 = merge_ints(vids['video1'])
        v2 = merge_ints(vids['video2'])
        fmt = lambda L: '; '.join(f"{s:.2f}-{e:.2f}s" for s, e in L)
        writer.writerow([gid, fmt(v1), fmt(v2)])


# -----------------------------------------------------------------------------
# Two-pass annotation using final global IDs
# -----------------------------------------------------------------------------
def annotate_video(in_path, out_path, fps, bboxes):
    cap = cv2.VideoCapture(in_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        for bbox, gid in bboxes.get(frame_idx, []):
            x, y, w_, h_ = bbox
            cv2.rectangle(frame, (x, y), (x + w_, y + h_), (0, 255, 0), 2)
            cv2.putText(frame, f"ID{gid}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        writer.write(frame)
    cap.release()
    writer.release()


# build per-frame dicts
per1 = defaultdict(list)
per2 = defaultdict(list)
for t, lst in track_bboxes.items():
    for fnum, bbox in lst:
        if t in tracks_video1:
            per1[fnum].append((bbox, t.id))
        else:
            per2[fnum].append((bbox, t.id))

annotate_video(args.video1, args.output1, fps1, per1)
annotate_video(args.video2, args.output2, fps2, per2)

print("Done! Videos annotated with consistent IDs and CSV saved.")
