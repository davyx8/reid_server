from fastapi import FastAPI, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
import shutil, uuid, os
from .models import engine, Session, Video, Person, Appearance, Anomaly
from .worker import worker_loop  # ensures the worker thread starts
from sqlalchemy import select

app = FastAPI(title="Multi‑Video Person Tracking")

STORAGE = "storage/videos"

@app.post("/videos")
async def upload_video(file: UploadFile, bg: BackgroundTasks):
    uid = str(uuid.uuid4())
    path = os.path.join(STORAGE, f"video_{uid}.mp4")
    # save file
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    with Session(engine) as s:
        v = Video(path=path)
        s.add(v); s.commit(); s.refresh(v)
        vid = v.id
    # worker picks it up automatically
    return {"video_id": vid}

@app.get("/persons")
def list_persons():
    with Session(engine) as s:
        persons = []
        for p in s.scalars(select(Person)).all():
            apps = s.scalars(select(Appearance)
                             .where(Appearance.person_id == p.id)
                             .order_by(Appearance.start_ts)).all()
            persons.append({
                "person_id": p.id,
                "appearances": [
                    {"video_id": a.video_id,
                     "start_ts": a.start_ts,
                     "end_ts": a.end_ts}
                     for a in apps]
            })
        return persons

@app.get("/persons/{pid}/anomalies")
def person_anoms(pid: int):
    with Session(engine) as s:
        a = s.scalars(select(Anomaly)
                      .where(Anomaly.person_id == pid)
                      .order_by(Anomaly.timestamp)).all()
        return [{"video_id": x.video_id,
                 "timestamp": x.timestamp,
                 "type": x.type,
                 "detail": x.detail} for x in a]

@app.get("/videos/{vid}/annotated")
def get_annotated(vid: int):
    with Session(engine) as s:
        v = s.get(Video, vid)
        if not v or not v.done:
            return JSONResponse({"status":"processing"}, status_code=202)
    return FileResponse(v.out_path, media_type="video/mp4")
