# app/track_impl.py
"""
Self‑contained tracking + anomaly detector.

Usage:
    tracks, track_bboxes, fps = run_tracking("/path/to/video.mp4")

After you have matched each Track to a global person_id, call
    out = f"storage/outputs/annotated_{video_id}.mp4"
    write_annotated_video(
        in_path     = video_path,
        out_path    = out,
        fps         = fps,
        track_bboxes= track_bboxes,         # from run_tracking
        anomalies_map = build_anomaly_map(tracks, track_bboxes, fps)
    )
"""

from __future__ import annotations
import cv2, numpy as np, torch, torch.nn as nn
from ultralytics import YOLO
from collections import defaultdict
from typing import Dict, List, Tuple
import torchreid, os

# --------------------------------------------------------------------------
# 0.  Global, memoised models
# --------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _init_models():
    detect = YOLO("yolov8n.pt")                 # person & object detector
    reid   = torchreid.models.build_model(
        name="osnet_x0_25", num_classes=1000,
        loss="softmax", pretrained=True)
    reid.classifier = nn.Identity()
    reid.eval().to(DEVICE)
    return detect, reid

_DETECT_MODEL, _REID_MODEL = _init_models()

# --------------------------------------------------------------------------
# 1.  Helpers: detection and feature extraction
# --------------------------------------------------------------------------
def detect_frame(frame) -> Tuple[List[Tuple[int,int,int,int]], List[set[str]]]:
    """
    Returns:
        pboxes     – list[(x,y,w,h)]
        pobj_sets  – list[set(class_names)] objects overlapping each person box
    """
    res = _DETECT_MODEL(frame[..., ::-1])  # BGR→RGB
    boxes = res[0].boxes
    persons, objects = [], []
    for box, cls, conf in zip(boxes.xyxy.cpu().numpy(),
                              boxes.cls.cpu().numpy(),
                              boxes.conf.cpu().numpy()):
        (x1,y1,x2,y2) = map(int, box)
        if cls == 0 and conf >= 0.35:      # class 0 = "person"
            persons.append(((x1,y1,x2,y2), conf))
        elif cls != 0 and conf >= 0.70:    # stricter for objects
            objects.append(((x1,y1,x2,y2), int(cls)))

    pboxes, pobj_sets = [], []
    for (x1,y1,x2,y2), _ in persons:
        area = (x2-x1)*(y2-y1)
        oset = set()
        for (ox1,oy1,ox2,oy2), ocls in objects:
            ix1,iy1 = max(x1,ox1), max(y1,oy1)
            ix2,iy2 = min(x2,ox2), min(y2,oy2)
            inter   = max(0,ix2-ix1)*max(0,iy2-iy1)
            if inter/float(area) >= 0.30:
                oset.add(res[0].names[ocls])
        pboxes.append((x1,y1,x2-x1,y2-y1))
        pobj_sets.append(oset)
    return pboxes, pobj_sets


def extract_reid_feature(bgr_crop: np.ndarray) -> np.ndarray:
    im = cv2.resize(bgr_crop, (128,256))[:, :, ::-1].astype(np.float32)/255.0
    mean = np.array([.485,.456,.406], np.float32)
    std  = np.array([.229,.224,.225], np.float32)
    im = (im-mean)/std
    ten = torch.from_numpy(im.transpose(2,0,1)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = _REID_MODEL(ten).squeeze(0).cpu().numpy()
    n = np.linalg.norm(feat)
    return feat/n if n>0 else feat


# --------------------------------------------------------------------------
# 2.  Track class (same anomaly logic you already tested)
# --------------------------------------------------------------------------
class Track:
    next_id = 1
    def __init__(self, bbox, feature, frame_idx, obj_set):
        self.local_id = Track.next_id; Track.next_id += 1
        self.bbox = bbox
        self.gid = -1
        self.feature_sum = feature.copy(); self.feature_count = 1
        self.start_frame = self.end_frame = frame_idx
        self.active = True; self.missed = 0

        # anomaly state
        self.base_feat_sum = feature.copy(); self.base_feat_count = 1
        self.obj_set = set(obj_set)
        self.anomalies: List[Tuple[int,str,str]] = []   # (frame, type, details)
        self.clothes_changed = False

    # ------------------------------------------------------------------
    def avg_feature(self):
        f = self.feature_sum / self.feature_count
        n = np.linalg.norm(f)
        return f/n if n>0 else f

    def mark_missed(self):
        self.missed += 1
        if self.missed > 30:
            self.active = False

    # ------------------------------------------------------------------
    def update(self, bbox, feature, frame_idx, obj_set):
        self.bbox = bbox
        self.end_frame = frame_idx
        self.missed = 0
        self.feature_sum += feature; self.feature_count += 1

        # ----- apparel change
        if not self.clothes_changed:
            base = self.base_feat_sum / self.base_feat_count
            sim  = float(np.dot(base, feature))
            if sim < 0.60:
                self.anomalies.append((frame_idx,"clothes",f"cos={sim:.2f}"))
                self.clothes_changed = True
        if not self.clothes_changed and self.base_feat_count < 20:
            self.base_feat_sum += feature; self.base_feat_count += 1

        # ----- object set change
        new_objs = set(obj_set)
        if new_objs != self.obj_set:
            diff = new_objs.symmetric_difference(self.obj_set)
            self.anomalies.append((frame_idx,"object",f"Δ={diff}"))
            self.obj_set = new_objs


# --------------------------------------------------------------------------
# 3.  Main function that **only processes ONE video**
# --------------------------------------------------------------------------
def run_tracking(video_path: str):
    """
    Returns:
        tracks        – list[Track]  (local ids, anomalies filled)
        track_bboxes  – dict[Track]->list[(frame_idx,bbox)]
        fps           – float
    """
    Track.next_id = 1                 # reset local counter
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    tracks : List[Track] = []
    track_bboxes : Dict[Track,List[Tuple[int,Tuple]]] = defaultdict(list)
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame_idx += 1

        pboxes, pobj_sets = detect_frame(frame)
        updated = []

        for bbox, obj_set in zip(pboxes, pobj_sets):
            x,y,w,h = bbox
            crop = frame[y:y+h, x:x+w]
            feat = extract_reid_feature(crop)

            # ---------- greedy IoU+cosine matching ----------
            best, best_iou = None, 0
            for t in tracks:
                if not t.active: continue
                tx,ty,tw,th = t.bbox
                ix1,iy1 = max(tx,x), max(ty,y)
                ix2,iy2 = min(tx+tw,x+w), min(ty+th,y+h)
                inter = max(0,ix2-ix1)*max(0,iy2-iy1)
                if inter == 0: continue
                iou = inter/float(tw*th + w*h - inter)
                if iou > 0.3:
                    sim = np.dot(t.avg_feature(), feat)
                    if sim>0.3 and iou>best_iou:
                        best_iou, best = iou, t

            if best:
                best.update(bbox, feat, frame_idx, obj_set)
                track = best
            else:
                track = Track(bbox, feat, frame_idx, obj_set)
                tracks.append(track)

            updated.append(track)
            track_bboxes[track].append((frame_idx, track.bbox))

        for t in tracks:
            if t not in updated: t.mark_missed()

    cap.release()
    return tracks, track_bboxes, fps


# --------------------------------------------------------------------------
# 4.  Utility to build anomaly map and final annotated video
# --------------------------------------------------------------------------
def build_anomaly_map(tracks, track_bboxes, fps) -> Dict[int,List]:
    """
    Returns:
        map frame_idx -> list[(bbox, person_id, type, detail)]
        NOTE: assumes Track.id has been overwritten with GLOBAL person_id.
    """
    amap = defaultdict(list)
    for t in tracks:
        for fr, typ, det in t.anomalies:
            # locate bbox in that frame
            for fnum, bbox in track_bboxes[t]:
                if fnum == fr:
                    amap[fnum].append((bbox, t.gid, typ, det))
                    # show caption for ~1 s (≈ fps frames)
                    for i in range(int(fps)):
                        amap[fnum+i].append((bbox, t.gid, typ, det))
                    break
    return amap


def write_annotated_video(in_path: str,
                          out_path: str,
                          fps: float,
                          track_bboxes: Dict,
                          anomalies_map: Dict[int,List]):
    # build per‑frame box dict with global IDs
    per_frame = defaultdict(list)
    for t, lst in track_bboxes.items():
        for f, bbox in lst:
            per_frame[f].append((bbox, t.gid))

    cap = cv2.VideoCapture(in_path)
    w,h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    four = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    wr = cv2.VideoWriter(out_path, four, fps, (w,h))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame_idx += 1

        # regular boxes (green)
        for bbox, pid in per_frame.get(frame_idx, []):
            x,y,w_,h_ = bbox
            cv2.rectangle(frame, (x,y), (x+w_, y+h_), (0,255,0), 2)
            cv2.putText(frame, f"ID{pid}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # anomaly captions (red)
        for bbox, pid, typ, det in anomalies_map.get(frame_idx, []):
            x,y,w_,h_ = bbox
            txt = f"{typ}:{det}"
            (tw,th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x, y+h_+2), (x+tw, y+h_+th+2), (0,0,0), cv2.FILLED)
            cv2.putText(frame, txt, (x, y+h_+th+2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        wr.write(frame)

    cap.release(); wr.release()
