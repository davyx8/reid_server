# app/worker.py
import threading, time, sqlalchemy as sa
from .models import engine, Video
from .processing import process_single_video

def worker_loop():
    while True:
        with sa.orm.Session(engine) as session:
            vid = session.query(Video).filter_by(done=False, processing=False).first()
            if vid:
                vid.processing = True
                session.commit()  # 🔸 commit so other workers skip it
                process_single_video(vid, session)
            else:
                time.sleep(2)

threading.Thread(target=worker_loop, daemon=True).start()
