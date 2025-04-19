from flask import Blueprint, request, redirect, url_for, session, render_template, flash, jsonify,Response
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


@user.route('/avatar/<uid>')
def get_user_avatar(uid):
    try:
        user = session['user_id']
        if not user or not user.Uheadshot:
            return '', 404  # 如果没有头像或用户不存在

        img_data = base64.b64decode(user.Uheadshot)
        return Response(img_data, mimetype='image/png')  # 或根据实际头像格式换成 image/jpeg

    except Exception as e:
        return '', 500


@user.route('/update_info', methods=['POST'])
def update_info():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '未登录'}), 401

    user_id = session['user_id']

    try:
        update_data = {
            'Uname': request.form.get('Uname'),
            'Ugender': request.form.get('Ugender'),
            'Uphone': request.form.get('Uphone'),
            'Upassword': request.form.get('Upassword') or None
        }

        if not all([update_data['Uname'], update_data['Ugender'], update_data['Uphone']]):
            return jsonify({'success': False, 'message': '必填字段不能为空'}), 400

        if len(update_data['Uphone']) != 11 or not update_data['Uphone'].isdigit():
            return jsonify({'success': False, 'message': '手机号格式错误'}), 400

        # 处理头像
        if 'Uheadshot' in request.files and request.files['Uheadshot'].filename != '':
            file = request.files['Uheadshot']
            if file and allowed_file(file.filename):
                filename = secure_filename(
                    f"{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.{file.filename.split('.')[-1]}"
                )
                file_path = os.path.join(AVATAR_BASE_PATH, filename)
                file.save(file_path)

                with open(file_path, "rb") as image_file:
                    # 直接保存为二进制数据，不做 base64 编码
                    update_data['Uheadshot'] = image_file.read()  # 保留为二进制数据

                # 之后直接传给 service 层处理

                print(f"[上传头像] Base64 字符长度: {len(update_data['Uheadshot'])}")


        # 调用 service 层
        u = User_operation()
        print("准备传给service层的头像数据是否存在：", bool(update_data.get('Uheadshot')))
        print("request.files:", request.files)  # 查看里面是否有 Uheadshot
        print("Uheadshot文件名：", request.files['Uheadshot'].filename if 'Uheadshot' in request.files else "无")

        result = u.update_info(
            Uid=user_id,
            Uname=update_data['Uname'],
            Uheadshot=update_data.get('Uheadshot'),
            Ugender=update_data['Ugender'],
            Uphone=update_data['Uphone'],
            Upassword=update_data['Upassword']
        )

        print('返回前端的数据：', result)

        # 返回 JSON 给前端 AJAX
        if result['code'] == 0:
            return jsonify({
                'success': True,
                'message': '信息更新成功',
                'newName': result['data']['Uname'],
                'newAvatar': url_for('user.get_user_avatar', uid=user_id)
            })
        else:
            return jsonify({'success': False, 'message': result['message']})

    except Exception as e:
        print(f"更新错误: {str(e)}")
        return jsonify({'success': False, 'message': '服务器异常，请联系管理员'}), 500



@user.route('/delete_account', methods=['POST'])
def delete_account():
    try:
        user_id = session.get('user_id')
        if not user_id:
            flash('用户未登录', 'error')
            return redirect(url_for('user.login'))

        u = User_operation()
        result = u.delete_user(user_id)

        if result.json['code'] != 0:
            flash(result.json['message'], 'error')
            return redirect(url_for('user.usercenter'))  # 删除失败，返回用户中心

        session.clear()
        flash('账号已成功删除', 'success')
        return redirect(url_for('main.homepage'))  # 删除成功，重定向到首页

    except Exception as e:
        flash('删除失败，请联系管理员', 'error')
        print(f"删除账号异常：{str(e)}")
        return redirect(url_for('user.usercenter'))  # 异常情况也返回用户中心

@user.route('/logout', methods=['POST'])
def logout():
    try:
        session.clear()
        return redirect(url_for('main.homepage'))
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

