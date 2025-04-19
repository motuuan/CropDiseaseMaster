import torch
from PIL import Image
from io import BytesIO
import numpy as np
from flask import jsonify
from Model.Crops import Crops
from ultralytics import YOLO
import torchvision.transforms as transforms
import re
from torch import nn
from torchvision import models

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.FloatTensor)

# 加载YOLOv8模型
try:
    yolo_model = YOLO("../Data/best.pt").to(device)
    yolo_model.model.float()
    print(f"YOLO model loaded on {device}")
except Exception as e:
    print(f"Error loading YOLO model: {str(e)}")
    yolo_model = None

img_size = 128


# 定义 SE 注意力模块
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# 增强版VGG模型
class BasicVGG(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        original_vgg = models.vgg16(pretrained=False)

        # 完全复现训练代码的特征层结构
        self.features = nn.Sequential(
            *list(original_vgg.features.children())[:3],
            SEBlock(64),
            *list(original_vgg.features.children())[3:7],
            SEBlock(128),
            *list(original_vgg.features.children())[7:14],
            SEBlock(256),
            original_vgg.features[14]
        )

        # 动态维度计算（必须保留）
        with torch.no_grad():
            test_input = torch.randn(1, 3, 128, 128)
            test_output = self.features(test_input)
            in_features = test_output.view(-1).shape[0]

        # 完全复现训练代码的分类器
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class CropTypeModel:
    def __init__(self, model_path, num_classes):
        # 完整类别列表（包含作物和病害）
        self.crop_mapping = {
            'Corn': '玉米',
            'Potato': '马铃薯',
            'Rice': '水稻',
            'Wheat': '小麦',
            'sugarcane': '甘蔗'
        }

        self.full_class_names = ['Corn___Common_Rust', 'Corn___Gray_Leaf_Spot', 'Corn___Healthy',
                                 'Corn___Northern_Leaf_Blight', 'Potato___Early_Blight', 'Potato___Healthy',
                                 'Potato___Late_Blight', 'Rice___Brown_Spot', 'Rice___Healthy',
                                 'Rice___Leaf_Blast', 'Rice___Neck_Blast', 'Wheat___Brown_Rust',
                                 'Wheat___Healthy', 'Wheat___Yellow_Rust', 'sugarcane___Bacterial Blight',
                                 'sugarcane___Healthy', 'sugarcane___Red Rot']

        try:
            self.model = BasicVGG(num_classes=num_classes)
            state_dict = torch.load(model_path, map_location='cpu')

            # ---------- 添加这段代码：修正模型权重的键名 ----------
            def fix_state_dict_keys(state_dict):
                """去掉多余的 'features.' 前缀（如 'features.features.' -> 'features.'）"""
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("features.features."):
                        new_key = k.replace("features.features.", "features.")
                    else:
                        new_key = k
                    new_state_dict[new_key] = v
                return new_state_dict

            # 应用键名修复
            state_dict = fix_state_dict_keys(state_dict)

            # 获取模型结构中允许的 key
            model_keys = set(self.model.state_dict().keys())

            # 自动过滤 + 修正后的最终 state_dict
            filtered_dict = {k: v for k, v in state_dict.items() if k in model_keys}

            # 强制维度验证 -----------------------------------------------------
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 128, 128)
                features_out = self.model.features(dummy_input)
                actual_in_features = features_out.view(1, -1).shape[1]

                expected_in_features = filtered_dict['classifier.0.weight'].shape[1]

                assert actual_in_features == expected_in_features, \
                    f"维度不匹配！模型实际输入: {actual_in_features}, 参数期望输入: {expected_in_features}"

            # 加载参数
            self.model.load_state_dict(filtered_dict, strict=True)

            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            self.model = self.model.to(device).eval()
            print("Crop classification model loaded successfully")
        except Exception as e:
            print(f"Error loading crop model: {str(e)}")
            raise

    def predict(self, image):
        try:
            image_tensor = self.transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = self.model(image_tensor)
            _, predicted_class = torch.max(output, 1)
            # 返回作物类型（分割第一部分）
            full_name = self.full_class_names[predicted_class.item()]
            en_crop = full_name.split('___')[0]
            # 转换为中文作物名称
            return self.crop_mapping.get(en_crop, en_crop)
        except Exception as e:
            return "未知作物"


try:
    crop_type_model = CropTypeModel(
        model_path="D:/pyproj/GraduateProject/Data/SE7_vgg.pth",
        num_classes=17
    )
except Exception as e:
    print(f"Crop model initialization failed: {str(e)}")
    crop_type_model = None


class CropsService:
    # YOLO标签到中文映射
    YOLO_DISEASE_MAPPING = {
        # Corn
        'Corn___Common_Rust': '玉米普通锈病',
        'Corn___Gray_Leaf_Spot': '玉米灰斑病',
        'Corn___Northern_Leaf_Blight': '玉米大斑病',
        'Corn___Healthy': '健康',
        # Potato
        'Potato___Early_Blight': '马铃薯早疫病',
        'Potato___Late_Blight': '马铃薯晚疫病',
        'Potato___Healthy': '健康',
        # Rice
        'Rice___Brown_Spot': '水稻胡麻斑病',
        'Rice___Leaf_Blast': '水稻稻瘟病',
        'Rice___Neck_Blast': '水稻穗颈瘟',
        'Rice___Healthy': '健康',
        # Wheat
        'Wheat___Brown_Rust': '小麦杆锈病',
        'Wheat___Yellow_Rust': '小麦条锈病',
        'Wheat___Healthy': '健康',
        # Sugarcane
        'sugarcane__Bacterial Blight': '甘蔗白条病',
        'sugarcane__Red Rot': '甘蔗赤腐病',
        'sugarcane__Healthy': '健康',
        # Healthy
        'Healthy': '健康'
    }

    @staticmethod
    def validate_image(image_file):
        try:
            image_data = image_file.read()
            image = Image.open(BytesIO(image_data))
            image.verify()
            image = Image.open(BytesIO(image_data)).convert("RGB")
            return image
        except Exception as e:
            print(f"Image validation failed: {e}")
            return None

    @staticmethod
    def process_yolo_results(results):
        if not results or not results[0].boxes:
            return []

        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()
        labels = results[0].names

        return [{
            'class_name': labels[int(class_ids[i])],
            'confidence': float(scores[i]),
            'bbox': boxes[i].tolist()
        } for i in range(len(boxes))]

    @staticmethod
    def output_results(image_file):
        try:
            image = CropsService.validate_image(image_file)
            if not image:
                return jsonify({'error': '无效的图片文件'}), 400

            # 1. VGG识别作物类型
            crop_class = "未知作物"
            if crop_type_model:
                try:
                    crop_class = crop_type_model.predict(image)
                    print(f"作物识别结果: {crop_class}")  # 调试输出
                except Exception as e:
                    print(f"作物识别错误: {str(e)}")

            # 2. YOLO病害检测
            disease_data = []
            if yolo_model and crop_class != "未知作物":
                try:
                    conf_threshold = 0.3
                    iou_threshold = 0.45

                    yolo_results = yolo_model.predict(
                        np.array(image),
                        imgsz=320,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        device=device,
                        verbose=False,
                        agnostic_nms=True
                    )

                    raw_results = CropsService.process_yolo_results(yolo_results)

                    # 只保留和当前作物相关的病害结果
                    filtered = [res for res in raw_results
                                if CropsService.is_related_disease(crop_class, res['class_name'])]

                    if filtered:
                        # 找出置信度最高的一个病害结果
                        best_result = max(filtered, key=lambda x: x['confidence'])

                        original_name = best_result['class_name']
                        zh_name = CropsService.YOLO_DISEASE_MAPPING.get(original_name, original_name)

                        record = Crops.query.filter(
                            Crops.Cclass == crop_class,
                            Crops.Cdisaster == zh_name
                        ).first()

                        if record:
                            disease_data.append({
                                'Cdisaster': record.Cdisaster,
                                'Csolution': record.Csolution
                            })
                        else:
                            disease_data.append({
                                'Cdisaster': zh_name,
                                'Csolution': "暂无防治方案"
                            })
                    else:
                        print("未检测到相关病害")

                except Exception as e:
                    print(f"病害检测错误: {str(e)}")

            # 构造响应数据
            cdisaster = []
            csolution = []
            for item in disease_data:
                cdisaster.append(item['Cdisaster'])
                csolution.append(item['Csolution'])

            response = {
                'Cclass': crop_class,
                'Cdisaster': '，'.join(cdisaster) if cdisaster else '-',
                'Csolution': '，'.join(csolution) if csolution else '-'
            }

            return jsonify(response)
        except Exception as e:
            print(f"处理流程异常: {str(e)}")
            return jsonify({'error': f"处理失败: {str(e)}"}), 500

    @staticmethod
    def is_related_disease(crop_type, disease_name):
        disease_mapping = {
            '玉米': ['Common_Rust', 'Gray_Leaf_Spot', 'Healthy', 'Northern_Leaf_Blight'],
            '马铃薯': ['Early_Blight', 'Healthy', 'Late_Blight'],
            '水稻': ['Brown_Spot', 'Healthy', 'Leaf_Blast', 'Neck_Blast'],
            '小麦': ['Brown_Rust', 'Healthy', 'Yellow_Rust'],
            '甘蔗': ['Bacterial Blight', 'Healthy', 'Red Rot']
        }
        return any(
            keyword in disease_name
            for keyword in disease_mapping.get(crop_type, [])
        )