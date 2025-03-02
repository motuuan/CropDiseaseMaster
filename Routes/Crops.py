from flask import Blueprint, request, jsonify
from Service.Crops import CropsService
from Model.Crops import Crops

# 创建Blueprint
crops_routes = Blueprint('crops_routes', __name__)

# 搜索病害的路由
@crops_routes.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '')

    if query:
        crops = Crops.query.filter(Crops.Cdisaster.like(f"%{query}%")).all()
    else:
        crops = []

    # 返回JSON数据给前端，避免重新加载页面
    result = [crop.to_dict() for crop in crops]

    return jsonify(result)


# 作物与病害识别的路由
@crops_routes.route('/api/recognize', methods=['POST'])
def recognize():
    if "image_file" not in request.files:
        return jsonify({"error": "No file part"}), 400  # 返回错误响应

    file = request.files["image_file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400  # 返回错误响应

    try:
        # 处理上传的图片
        result = CropsService.output_results(file)
        if result is None:
            return jsonify({"error": "No results found"}), 400  # 处理没有返回的情况

        return result  # 返回识别结果
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": f"Failed to process the image: {str(e)}"}), 500  # 错误处理

