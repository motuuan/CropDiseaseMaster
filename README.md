---

# 🌾 作物病害通：作物健康识别系统

本项目是一个基于 **Flask + YOLOv8 + VGG** 构建的作物健康识别系统，融合了深度学习图像识别技术与前后端开发能力，能够实现农作物种类自动识别及病害检测，并支持部分病害信息的检索与查看，旨在为农业提供智能化、低门槛的辅助诊断工具。

---

## 📌 项目亮点

- ✅ **双模型识别架构**：结合 VGG 模型进行作物种类识别，YOLOv8 模型用于病害检测，实现分类与定位的协同识别。
- ✅ **清晰的 MVC 架构**：采用模型-视图-控制器架构，模块划分明确，结构清晰，方便维护和扩展。
- ✅ **完整前后端功能**：具备美观直观的用户界面及稳定的数据处理逻辑，用户上传图像后可立即查看识别结果。
- ✅ **数据库支撑**：使用 MySQL 管理作物、病害及识别记录信息，实现数据的持久化与查询。
- ✅ **可本地部署体验**：项目可在本地一键运行，体验完整识别流程，未来可扩展至云端平台部署。

---

## 🧠 技术栈

| 模块     | 技术选型                         |
|----------|----------------------------------|
| 后端     | Python, Flask                    |
| 前端     | HTML, CSS, JavaScript            |
| 数据库   | MySQL                            |
| 模型     | YOLOv8（Ultralytics）, 自定义VGG |
| 部署方式 | 本地运行（支持未来部署）         |

---

## 🚀 快速开始（本地运行）

### 1. 克隆项目

```bash
git clone https://github.com/motuuan/CropDiseaseMaster.git
cd CropDiseaseMaster
````

### 2. 安装依赖

请确保你的环境中已安装 Python 3.8 及以上版本，然后执行：

```bash
pip install -r requirements.txt
```

### 3. 🗂 配置数据库（可选）

本项目提供了数据库结构 SQL 文件，位于 `database/crop_disease.sql`。
数据库内数据包含用户头像、病害详情图像等，由于文件大小限制，SQL文件仅包含结构，如需要相关数据，请联系我。

可使用以下命令导入：

```bash
mysql -u root -p your_database_name < database/crop_disease.sql
```
或使用图形化工具（如 Navicat、DataGrip）打开 crop_disease.sql 文件执行。

⚠️ 请确保提前创建数据库，并在 app.py 或配置文件中正确填写连接信息。

### 4. 启动服务

```bash
python app.py
```

运行成功后，在浏览器访问：

```
http://localhost:5000
```

---

## 📁 项目结构说明

```
GraduateProject/
│
├── static/                 # 静态资源（样式、脚本、图像）
├── templates/         # HTML 模板文件
├── database/          # 数据库文件
├── Data/                  # 模型文件（YOLOv8、VGG）
├── Models/             # 数据库 ORM 模型
├── Service/              # 业务逻辑层
├── Routes/              # 路由与视图控制器
├── app.py                # 项目主入口
└── requirements.txt    # 项目依赖列表
```

---

## 🎯 项目目标与背景

本项目结合人工智能与农业应用场景，通过深度学习与Web开发实现农作物图像智能识别，降低农户识别门槛，提升农业管理效率。

未来计划：

* 支持移动端适配与云端部署
* 扩展更多病害种类与模型精度优化
* 接入天气与种植建议等功能模块

---

## 🙋‍♀️ 开发者

作者：[motuuan](https://github.com/motuuan)
如有问题或建议，欢迎提交 Issue 或私信交流。

---

## 📃 License

MIT License
欢迎学习和使用，转载请注明原项目链接。

