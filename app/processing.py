# app/processing.py  (excerpt)
import faiss, numpy as np, pickle
from sqlalchemy.orm import Session
from .models import Person, Appearance, Anomaly
from .track_impl import build_anomaly_map, run_tracking, write_annotated_video
from sqlalchemy import exists, and_
from .models import Appearance   # already imported elsewhere

def intervals_overlap(a_start, a_end, b_start, b_end):
    return not (a_end < b_start or a_start > b_end)

class PersonIndex:
    """A Faiss index that lives on disk and mirrors the Person table."""
    D = 512

    def __init__(self, session: Session):
        self.session = session
        self._rebuild()

    def _rebuild(self):
        feats, ids = [], []
        for p in self.session.query(Person).all():
            v = p.get_feat()
            feats.append(v)
            ids.append(p.id)
        if feats:
            xb = np.stack(feats).astype('float32')
            self.index = faiss.IndexFlatIP(self.D)  # cosine (vectors already L2‑normed)
            self.index.add(xb)
            self.ids = np.array(ids)
        else:
            self.index = faiss.IndexFlatIP(self.D)
            self.ids = np.array([], dtype=int)

    def add_person(self, feat: np.ndarray, person_id: int):
        self.index.add(feat[None, :].astype('float32'))
        self.ids = np.append(self.ids, person_id)

    def match(self, feat: np.ndarray, thr=0.8):
        if self.index.ntotal == 0:
            return None, None
        D, I = self.index.search(feat[None, :].astype('float32'), 1)
        if D[0, 0] >= thr:
            return int(self.ids[I[0, 0]]), float(D[0, 0])
        return None, float(D[0, 0])


# processing.py  (excerpt)

def group_local_tracks(tracks, cos_thr=0.80):
    """
    Greedy grouping identical to original `match_track()`.
    Returns a list of lists; each inner list is one provisional person.
    """
    groups = []  # list[list[Track]]
    for t in tracks:
        feat = t.avg_feature()
        placed = False
        for g in groups:
            gf = np.mean([x.avg_feature() for x in g], axis=0)
            sim = float(np.dot(gf, feat))
            # disallow if *any* member overlaps in time
            overlap = any(not (t.end_frame < o.start_frame or
                               t.start_frame > o.end_frame) for o in g)
            if sim >= cos_thr and not overlap:
                g.append(t);
                placed = True;
                break
        if not placed:
            groups.append([t])
    return groups


def process_single_video(video_row, session):
    tracks, track_bboxes, fps = run_tracking(video_row.path)

    # 🔸 NEW: collapse track fragments first
    groups = group_local_tracks(tracks, cos_thr=0.80)

    pidx = PersonIndex(session)

    for g in groups:
        # average feature for the whole person



        mean_feat = np.mean([t.avg_feature() for t in g], axis=0)
        gid, _ = pidx.match(mean_feat)
        # ░░░ 1️⃣  TEMPORAL‑EXCLUSIVITY GUARD ░░░
        if gid is not None:
            # convert current group’s time span to seconds
            g_start = min(t.start_frame for t in g) / fps
            g_end = max(t.end_frame for t in g) / fps

            conflict = session.query(
                exists().where(
                    and_(
                        Appearance.person_id == gid,
                        Appearance.video_id == video_row.id,
                        Appearance.start_ts <= g_end,
                        Appearance.end_ts >= g_start
                    )
                )
            ).scalar()

            if conflict:
                gid = None  # ←  pretend it was “no match”
        # ░░░  END GUARD ░░░
        if gid is None:  # create a new global person
            p = Person()
            p.set_feat(mean_feat)
            session.add(p)
            session.flush()
            gid = p.id
            pidx.add_person(mean_feat, gid)
        else:  # update existing centroid
            p = session.get(Person, gid)
            new_feat = (p.get_feat() + mean_feat)
            new_feat /= np.linalg.norm(new_feat)
            p.set_feat(new_feat)

        # assign the global ID to *all* tracks in the group
        for t in g:
            t.gid = gid
            # Appearance rows
            app = Appearance(person_id=gid, video_id=video_row.id,
                             start_ts=t.start_frame / fps,
                             end_ts=t.end_frame / fps)
            session.add(app)
            # Anomaly rows
            for fr, typ, det in t.anomalies:
                session.add(
                    Anomaly(person_id=gid, video_id=video_row.id,
                            timestamp=fr / fps, type=typ, detail=det)
                )

    # commit DB side effects
    video_row.done = True
    anomalies_map = build_anomaly_map(tracks, track_bboxes, fps)
    out_path = f"storage/outputs/annotated_{video_row.id}.mp4"
    write_annotated_video(video_row.path, out_path, fps,
                          track_bboxes, anomalies_map)
    video_row.out_path = out_path
    video_row.processing = False

    session.commit()
