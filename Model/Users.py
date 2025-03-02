import base64
from GP_config import db_init as db


class Users(db.Model):
    __tablename__ = 'users'
    Uid = db.Column(db.String(255), primary_key=True, nullable=False)
    Uname = db.Column(db.String(255),  nullable=False)
    Upassword = db.Column(db.String(255), nullable=False)
    Uheadshot = db.Column(db.LargeBinary, nullable=False)
    Ugender = db.Column(db.String(255), nullable=False)
    Uphone = db.Column(db.String(255), nullable=False)

    def to_dict(self):
        return {
            'Uid': self.Uid,
            'Uname': self.Uname,
            'Upassword': self.Upassword,
            'Uheadshot': base64.b64encode(self.Uheadshot).decode('utf-8'),
            'Ugender': self.Ugender,
            'Uphone': self.Uphone
        }