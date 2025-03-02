import warnings
warnings.filterwarnings("ignore")

from flask import request, render_template, redirect, url_for, session
from functools import wraps
from GP_config import app
from Routes.Main import main
from Routes.Users import user

# 注册蓝图
app.register_blueprint(main)  # 主页\其他功能
app.register_blueprint(user, url_prefix="/user")  # 用户相关

app.secret_key = 'doxjjarjeh3die'  # 密钥

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('user.login'))  # 未登录，重定向到登录页面
        return f(*args, **kwargs)  # 已登录，继续执行原函数
    return decorated_function

@app.route('/')
def home():
    return redirect(url_for('main.homepage'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

