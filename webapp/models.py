"""Persistent user and attendance models."""

from datetime import date, datetime, time

from . import db


class User(db.Model):
    __tablename__ = "users"

    student_id = db.Column(db.String(80), primary_key=True)
    name = db.Column(db.String(160), nullable=False, index=True)
    embedding = db.Column(db.LargeBinary, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    attendance = db.relationship(
        "Attendance",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=False,
    )


class Attendance(db.Model):
    __tablename__ = "attendance"
    __table_args__ = (
        db.UniqueConstraint("student_id", "attendance_date", name="uq_attendance_student_day"),
        db.Index("ix_attendance_date_first_seen", "attendance_date", "first_seen"),
    )

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(
        db.String(80),
        db.ForeignKey("users.student_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    attendance_date = db.Column(db.Date, nullable=False, default=date.today)
    first_seen = db.Column(db.Time, nullable=False, default=lambda: datetime.now().time().replace(microsecond=0))
    last_seen = db.Column(db.Time, nullable=False, default=lambda: datetime.now().time().replace(microsecond=0))
    user = db.relationship("User", back_populates="attendance")

    @property
    def first_seen_text(self):
        return self.first_seen.strftime("%H:%M:%S") if isinstance(self.first_seen, time) else str(self.first_seen)

    @property
    def last_seen_text(self):
        return self.last_seen.strftime("%H:%M:%S") if isinstance(self.last_seen, time) else str(self.last_seen)
