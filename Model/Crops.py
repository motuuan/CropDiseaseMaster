import base64
from GP_config import db_init as db


class Crops(db.Model):
    __tablename__ = 'crops'

    Cno = db.Column(db.Integer, primary_key=True, autoincrement=True)  # 作物序号
    Cclass = db.Column(db.String(255), nullable=False)  # 作物名称
    Cdisaster = db.Column(db.String(255), nullable=False)  # 病害名称
    Cdescription = db.Column(db.Text, nullable=False)  # 病害描述
    Csolution = db.Column(db.Text, nullable=False)  # 病害解决方案
    Cpicture = db.Column(db.LargeBinary, nullable=True)
    Csymptoms = db.Column(db.Text, nullable=False)  # 病害解决方案

    def to_dict(self):
        return {
            'Cno': self.Cno,
            'Cclass': self.Cclass,
            'Cdisaster': self.Cdisaster,
            'Cdescription': self.Cdescription,
            'Csolution': self.Csolution,
            'Cpicture': base64.b64encode(self.Cpicture).decode('utf-8'),
            'Csymptoms': self.Csymptoms
        }
