from PIL import Image
import io
import base64
from GP_config import db_init as db
from Model.Crops import Crops
from app import app


def process_image(image_path):
    img = Image.open(image_path)

    img = img.resize((1064, 692))

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")

    image_data = buffer.getvalue()

    return image_data


def update_picture_in_db(Cno, image_path):
    with app.app_context():
        crop = Crops.query.get(Cno)

        if crop:
            image_data = process_image(image_path)

            # 更新Cpicture字段
            crop.Cpicture = image_data

            # 提交更改到数据库
            db.session.commit()
            print(f"作物 {Cno} 的图片已成功更新！")
        else:
            print(f"没有找到Cno为 {Cno} 的作物记录！")


Cno = 8
image_path = 'C:/Users/30761/Pictures/Screenshots/屏幕截图 2025-02-24 214919.png'

update_picture_in_db(Cno, image_path)

