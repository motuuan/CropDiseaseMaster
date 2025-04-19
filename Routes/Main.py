from flask import Blueprint, render_template, redirect, url_for, jsonify, request, session
import base64
from GP_config import app
from Service.Users import user_service
from Routes.Crops import crops_routes
from Routes.Records import records_routes
from Model.Crops import Crops

main = Blueprint('main', __name__)
app.register_blueprint(crops_routes, url_prefix='/crops')
app.register_blueprint(records_routes, url_prefix='/records')

@main.route('/')
def homepage():
    user = None
    if 'user_id' in session:
        user = user_service.get_user_info(session['user_id'])  # 获取用户信息
    return render_template('homepage.html', user=user, logged_in=(user is not None))


@main.route('/recognition')
def recognition():
    user = None
    if 'user_id' in session:
        user = user_service.get_user_info(session['user_id'])  # 获取用户信息
    return render_template('recognition.html', user=user, logged_in=(user is not None))

@main.route('/document')
def document():
    user = None
    if 'user_id' in session:
        user = user_service.get_user_info(session['user_id'])  # 获取用户信息
    return render_template('document.html', user=user, logged_in=(user is not None))

@main.route('/record')
def record():
    user = None
    if 'user_id' in session:
        user = user_service.get_user_info(session['user_id'])  # 获取用户信息
    return render_template('record.html', user=user, logged_in=(user is not None))

@main.route('/recordchart')
def recordchart():
    user = None
    if 'user_id' in session:
        user = user_service.get_user_info(session['user_id'])  # 获取用户信息
    return render_template('recordchart.html', user=user, logged_in=(user is not None))


@main.route('/disease_detail/<int:crop_id>', methods=['GET'])
def disease_detail(crop_id):
    user = None
    if 'user_id' in session:
        user = user_service.get_user_info(session['user_id'])  # 获取用户信息
    try:
        crop = Crops.query.filter_by(Cno=crop_id).first()
        picture_base64 = base64.b64encode(crop.Cpicture).decode('utf-8')
        if crop:
            return render_template('details.html', user=user, crop=crop, picture=picture_base64)  # 渲染病害详情页面
        return jsonify({'message': '病害信息未找到'}), 404  # 如果没有找到对应病害信息，返回404
    except Exception as e:
        return jsonify({'error': f"Error fetching disease detail: {str(e)}"}), 500


@main.route('/usercenter')
def usercenter():
    user = None
    if 'user_id' in session:
        user = user_service.get_user_info(session['user_id'])  # 获取用户信息
    return render_template('usercenter.html', user=user, logged_in=(user is not None))


# 退出登录
@main.route('/logout')
def logout():
    session.pop('token', None)
    return redirect(url_for('main.homepage'))
