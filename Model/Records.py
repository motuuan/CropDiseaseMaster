import base64
from sqlalchemy import func
from GP_config import db_init as db
from datetime import datetime


class Records(db.Model):
    __tablename__ = 'records'
    Rno = db.Column(db.Integer, primary_key=True, autoincrement=True, nullable=False)
    R_Uid = db.Column(db.String(255), db.ForeignKey('users.Uid'), nullable=False)
    Rtime = db.Column(db.DateTime, default=func.now(), nullable=False)
    Rclass = db.Column(db.String(255), nullable=False)
    Rdisaster = db.Column(db.String(255), nullable=False)
    Rpicture = db.Column(db.LargeBinary, nullable=False)

    def to_dict(self):
        return {
            'Rno': self.Rno,
            'R_Uid': self.R_Uid,
            'Rtime': self.Rtime.strftime('%Y-%m-%d %H:%M:%S'),
            'Rclass': self.Rclass,
            'Rdisaster': self.Rdisaster,
            'Rpicture': base64.b64encode(self.Rpicture).decode('utf-8')
        }