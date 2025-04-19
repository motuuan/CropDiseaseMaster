from flask import Blueprint, request, jsonify
from Service.Records import RecordsService

records_routes = Blueprint("records_routes", __name__)

@records_routes.route('/getrecord', methods=['GET'])
def get_records():
    user_id = request.args.get("user_id")
    page = request.args.get("page", 1, type=int)

    print(f"收到请求: user_id={user_id}, page={page}")

    if not user_id:
        return jsonify({"error": "用户ID未提供"}), 400

    records, total_pages = RecordsService.get_user_records(user_id, page, per_page=5)

    print(f"返回记录数: {len(records)}, 总页数: {total_pages}")

    return jsonify({
        "records": records,
        "current_page": page,
        "total_pages": total_pages
    })

@records_routes.route('/getchartdata', methods=['GET'])
def get_chart_data():
    user_id = request.args.get("user_id")
    print(f"获取统计数据，请求的 user_id: {user_id}")  # 观察 user_id 是否获取正确

    if not user_id:
        return jsonify({"error": "用户ID未提供"}), 400  # 返回错误信息

    try:
        chart_data = RecordsService.get_chart_data(user_id)  # 假设这里获取统计数据
        print(f"查询到的统计数据: {chart_data}")  # 观察数据是否正确

        if not chart_data:
            return jsonify({"error": "没有找到统计数据"}), 404

        return jsonify(chart_data)

    except Exception as e:
        print(f"获取图表数据时发生错误: {str(e)}")  # 记录错误信息
        return jsonify({"error": "服务器内部错误"}), 500
