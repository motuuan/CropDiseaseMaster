from flask import jsonify
import base64
from PIL import Image
import io
from datetime import datetime
from GP_config import db_init as db
from Model.Records import Records
from sqlalchemy import func

class RecordsService:
    @staticmethod
    def save_record(user_id, image_file, crop_class, disease_name):
        try:
            img = Image.open(image_file)

            img = img.resize((1064, 692))

            buffer = io.BytesIO()
            img.save(buffer, format="PNG")  # 这里可以改成 "JPEG" 或其他格式
            image_data = buffer.getvalue()  # 获取二进制数据

            new_record = Records(
                R_Uid=user_id,
                Rtime=datetime.now(),
                Rclass=crop_class,
                Rdisaster=disease_name,
                Rpicture=image_data  # 存入数据库
            )

            db.session.add(new_record)
            db.session.commit()

            return {"message": "记录保存成功"}
        except Exception as e:
            db.session.rollback()
            return {"error": f"记录保存失败: {str(e)}"}

    @staticmethod
    def get_user_records(user_id, page, per_page):
        try:
            # 查询该用户的识别记录
            query = Records.query.filter_by(R_Uid=user_id)
            total_records = query.count()
            total_pages = (total_records + per_page - 1) // per_page  # 计算总页数

            records = query.order_by(Records.Rtime.desc()).paginate(page=page, per_page=per_page, error_out=False)

            result = []
            for record in records.items:
                result.append({
                    "image": base64.b64encode(record.Rpicture).decode("utf-8"),
                    "crop_name": record.Rclass,
                    "disease_name": record.Rdisaster,
                    "time": record.Rtime.strftime("%Y-%m-%d %H:%M:%S")
                })

            return result, total_pages
        except Exception as e:
            print(f"查询记录出错: {e}")
            return [], 0

    @staticmethod
    def get_chart_data(user_id):
        # 查询识别记录总数
        total_count = db.session.query(func.count(Records.R_Uid)).filter_by(R_Uid=user_id).scalar()

        # 查询近7天每日的识别次数
        date_records = (
            db.session.query(func.date(Records.Rtime), func.count(Records.Rno))
            .filter(Records.R_Uid == user_id)
            .group_by(func.date(Records.Rtime))
            .order_by(func.date(Records.Rtime))
            .all()
        )
        date_labels = [str(item[0]) for item in date_records]
        date_counts = [item[1] for item in date_records]

        # 统计作物类别占比（五种作物）
        crop_counts = (
            db.session.query(Records.Rclass, func.count(Records.Rno))
            .filter(Records.R_Uid == user_id)
            .group_by(Records.Rclass)
            .all()
        )
        crop_labels = [item[0] for item in crop_counts]
        crop_values = [item[1] for item in crop_counts]

        return {
            "total_count": total_count,
            "date_labels": date_labels,
            "date_counts": date_counts,
            "crop_labels": crop_labels,
            "crop_values": crop_values
        }