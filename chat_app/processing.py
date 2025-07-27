# app/processing.py  (excerpt)
import faiss, numpy as np, pickle
from sqlalchemy.orm import Session
from .models import Person, Appearance, Anomaly
from .track_impl import build_anomaly_map


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
            self.index = faiss.IndexFlatIP(self.D)   # cosine (vectors already L2‑normed)
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

def process_single_video(video_row, session):
    """
    1. Runs detection / tracking / anomaly logic (your full script).
    2. For every finished Track, performs global matching against PersonIndex.
    3. Inserts / updates Person, Appearance, Anomaly rows.
    4. Writes annotated video & updates Video.done/out_path.
    """
    from .track_impl import run_tracking,write_annotated_video   # your existing logic isolated into a fn
    tracks, track_bboxes, fps = run_tracking(video_row.path)
    # initialise / refresh index
    pidx = PersonIndex(session)

    for t in tracks:
        # 1) global match
        gid, sim = pidx.match(t.avg_feature())
        if gid is None:
            # make a new Person
            p = Person()
            p.set_feat(t.avg_feature())
            session.add(p)
            session.flush()  # assigns id
            gid = p.id
            pidx.add_person(t.avg_feature(), gid)
        else:
            p = session.get(Person, gid)
            # optional: update running mean feature
            new_feat = (p.get_feat() + t.avg_feature())
            new_feat /= np.linalg.norm(new_feat)
            p.set_feat(new_feat)

        # 2) appearance record
        app = Appearance(person_id=gid,
                         video_id=video_row.id,
                         start_ts=t.start_frame / fps,
                         end_ts=t.end_frame / fps)
        session.add(app)

        # 3) anomalies
        for fr, typ, det in t.anomalies:
            session.add(
                Anomaly(person_id=gid,
                        video_id=video_row.id,
                        timestamp=fr / fps,
                        type=typ,
                        detail=det)
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
