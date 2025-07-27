# app/models.py
from sqlalchemy import (Column, Integer, Float, String, Boolean,
                        ForeignKey, LargeBinary, create_engine, UniqueConstraint)
from sqlalchemy.orm import declarative_base, relationship, Session
import numpy as np
import io, pickle

Base = declarative_base()

class Video(Base):
    __tablename__ = "videos"
    id       = Column(Integer, primary_key=True)
    path     = Column(String, nullable=False)
    fps      = Column(Float)
    done     = Column(Boolean, default=False)
    out_path = Column(String)      # annotated video


    processing = Column(Boolean, default=False)  # <-- NEW

class Person(Base):
    __tablename__ = "persons"
    id       = Column(Integer, primary_key=True)
    feat     = Column(LargeBinary)      # mean 512‑D feature (L2 normalised)

    def get_feat(self):
        return np.frombuffer(self.feat, dtype=np.float32)

    def set_feat(self, vec: np.ndarray):
        self.feat = vec.astype(np.float32).tobytes()

class Appearance(Base):
    __tablename__ = "appearances"
    id        = Column(Integer, primary_key=True)
    person_id = Column(Integer, ForeignKey("persons.id"))
    video_id  = Column(Integer, ForeignKey("videos.id"))
    start_ts  = Column(Float)   # seconds
    end_ts    = Column(Float)

    person = relationship("Person", backref="appearances")
    video  = relationship("Video", backref="appearances")
    __table_args__ = (UniqueConstraint('person_id', 'video_id', 'start_ts', name='_uniq_app'),)

class Anomaly(Base):
    __tablename__ = "anomalies"
    id        = Column(Integer, primary_key=True)
    person_id = Column(Integer, ForeignKey("persons.id"))
    video_id  = Column(Integer, ForeignKey("videos.id"))
    timestamp = Column(Float)   # seconds
    type      = Column(String)  # "clothes" | "object"
    detail    = Column(String)

engine = create_engine("sqlite:///db.sqlite", echo=False, future=True)
Base.metadata.create_all(engine)
