from . import db
from flask_login import UserMixin
from sqlalchemy.sql import func


class Recording(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    date = db.Column(db.DateTime(timezone=True), default=func.now())
    audio_blob = db.Column(db.LargeBinary)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    performance_report = db.relationship(
        "PerformanceReport", uselist=False, back_populates="recording"
    )

    def __repr__(self):
        return f"Recording #{self.id} ({self.name}, {self.date})"


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True)
    username = db.Column(db.String(100))
    password = db.Column(db.String(100))
    recordings = db.relationship("Recording")

    def __repr__(self):
        return f"User {self.username} ({self.email}))"


class PerformanceReport(db.Model):
    id = db.Column(db.Integer, db.ForeignKey("recording.id"), primary_key=True)

    duration = db.Column(db.Float)
    speech_duration = db.Column(db.Float)
    ratio_speech_time = db.Column(db.Float)
    intensity_mean = db.Column(db.Float)
    intensity_std = db.Column(db.Float)
    intensity_max = db.Column(db.Float)
    pitch_mean = db.Column(db.Float)
    pitch_std = db.Column(db.Float)
    pitch_min = db.Column(db.Float)
    pitch_max = db.Column(db.Float)

    recording = db.relationship("Recording", back_populates="performance_report")

    def __repr__(self):
        return f"PerformanceReport for Recording#{self.id})"
