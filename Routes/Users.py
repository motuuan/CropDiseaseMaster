from flask import Blueprint, request, redirect, url_for, session, render_template, flash, jsonify
from flask_login import logout_user, current_user
import os
from datetime import datetime
from werkzeug.utils import secure_filename
from Service.Users import User_operation
import base64

user = Blueprint('user', __name__)
AVATAR_BASE_PATH = 'static/headshots/'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def login_required(f):
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('请先登录！', 'error')
            return redirect(url_for('user.login'))
        return f(*args, **kwargs)
    return decorated_function


@user.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'POST':
        data = request.form
        Uid = data['Uid']
        Upassword = data['Upassword']
        Uname = data['Uname']
        Ugender = data['Ugender']
        Uphone = data['Uphone']
        confirmpassword = data['confirmpassword']

        if Upassword != confirmpassword:
            flash('确认密码与密码不一致，请重新输入!', 'error')
            return redirect(url_for('user.register'))

        Uheadshot = None
        if Uheadshot is None:
            with open('static/headshots/initialhead.png', 'rb') as f:
                Uheadshot = f.read()
        if 'Uheadshot' in request.files and request.files['Uheadshot'].filename != '':
            file = request.files['Uheadshot']
            if file and allowed_file(file.filename):
                file_data = file.read()  # 读取文件数据
                Uheadshot = file_data
            else:
                flash('上传的文件类型不被允许', 'error')
                return redirect(url_for('user.register'))

        u = User_operation()
        result = u.register(Uid, Uname, Uheadshot, Ugender, Uphone, Upassword)

        print(f"Debug: Register result: {result.json}")

        if result.json['code'] == 0:
            flash(result.json['message'], 'success')
            return redirect(url_for('user.login'))
        else:
            flash(result.json['message'], 'error')
            return redirect(url_for('user.register'))

    return render_template('register.html')

@user.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        data = request.form
        Uid = data.get('Uid')
        Upassword = data.get('Upassword')
        u = User_operation()
        result = u.login(Uid, Upassword)

        if result.json['code'] == 0:
            session['user_id'] = Uid
            print("✅ 用户已登录，session:", session)  # 终端调试
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'message': result.json['message']})

    return render_template('login.html')


@user.route('/forgot_password', methods=['GET'])
def forgot_password():
    Uid = request.args.get('Uid')
    if not Uid:
        return jsonify({'code': -1, 'message': '用户名不能为空'})

    u = User_operation()
    result = u.forget_password(Uid)
    return result


@user.route('/update_info', methods=['POST'])
def update_info():
    if 'user_id' not in session:
        flash('请先登录！', 'error')
        return redirect(url_for('user.login'))

    user_id = session['user_id']

    try:
        # 获取表单数据
        update_data = {
            'Uname': request.form.get('Uname'),
            'Ugender': request.form.get('Ugender'),
            'Uphone': request.form.get('Uphone'),
            'Upassword': request.form.get('Upassword') or None  # 密码可选
        }

        # 基础验证
        if not all([update_data['Uname'], update_data['Ugender'], update_data['Uphone']]):
            flash('必填字段不能为空', 'error')
            return redirect(url_for('usercenter'))  # 失败后重定向到个人中心

        if len(update_data['Uphone']) != 11 or not update_data['Uphone'].isdigit():
            flash('手机号格式不正确', 'error')
            return redirect(url_for('usercenter'))

        # 处理头像上传
        if 'Uheadshot' in request.files and request.files['Uheadshot'].filename != '':
            file = request.files['Uheadshot']
            if file and allowed_file(file.filename):
                filename = secure_filename(
                    f"{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.{file.filename.split('.')[-1]}"
                )
                file_path = os.path.join(AVATAR_BASE_PATH, filename)
                file.save(file_path)

                # 将头像转为 base64 存储
                with open(file_path, "rb") as image_file:
                    update_data['Uheadshot'] = base64.b64encode(image_file.read()).decode('utf-8')

        # 更新用户信息
        u = User_operation()
        result = u.update_info(
            Uid=user_id,
            Uname=update_data['Uname'],
            Uheadshot=update_data.get('Uheadshot'),
            Ugender=update_data['Ugender'],
            Uphone=update_data['Uphone'],
            Upassword=update_data['Upassword']
        )

        if result.json['code'] == 0:
            flash('信息更新成功！', 'success')
        else:
            flash('更新失败，请重试', 'error')

    except Exception as e:
        flash('服务器错误，请联系管理员', 'error')
        print(f"更新错误: {str(e)}")

    return redirect(url_for('main.usercenter'))  # 修改后重定向回个人中心页面



@user.route('/delete_account', methods=['POST'])
def delete_account(current_user):
    try:
        u = User_operation()
        result = u.delete_user(current_user)

        if result.json['code'] != 0:
            return jsonify({'success': False, 'message': result.json['message']}), 400

        session.clear()
        return jsonify({
            'success': True,
            'message': '账号已删除',
            'redirect': url_for('user.login')
        }), 200

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@user.route('/logout', methods=['POST'])
def logout():
    try:
        session.clear()
        return jsonify({
            'success': True,
            'message': '已退出登录',
            'redirect': url_for('user.login')
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@user.route('/get_profile', methods=['GET'])
def get_profile():
    Uid = session.get('Uid')

    if not Uid:
        return jsonify({'success': False, 'message': '用户未登录'})

    u = User_operation()
    user_info = u.get_user_info(Uid)

    if user_info and user_info.Uheadshot:
        avatar_url = f"/user/avatar/{Uid}"  # 头像的 URL
    else:
        avatar_url = "/static/headshots/initialhead.png"

    return jsonify({'success': True, 'avatar': avatar_url})

