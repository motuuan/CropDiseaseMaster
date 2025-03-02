import torch
from PIL import Image
from io import BytesIO
import numpy as np
from flask import jsonify
from Model.Crops import Crops
from ultralytics import YOLO
import torchvision.transforms as transforms
import math
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


# 注意力机制模块
class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x)
        x = x * ca
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([max_pool, avg_pool], dim=1))
        return x * sa


class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


# 增强版VGG模型
class EnhancedVGG(nn.Module):
    def __init__(self, num_classes):
        super(EnhancedVGG, self).__init__()
        original_vgg = models.vgg16(pretrained=False)
        self.features = nn.Sequential()
        conv_layers = list(original_vgg.features.children())
        attention_positions = [24, 28, 30]

        for i, layer in enumerate(conv_layers):
            self.features.add_module(str(i), layer)
            if i in attention_positions:
                if i == 24:
                    self.features.add_module(f'cbam_{i}', CBAM(512))
                else:
                    self.features.add_module(f'eca_{i}', ECABlock(512))

        self._dummy_input = torch.zeros(1, 3, 128, 128)
        with torch.no_grad():
            self._dummy_output = self.features(self._dummy_input)
        num_features = self._dummy_output.numel()

        self.classifier = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
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
            self.model = EnhancedVGG(num_classes=num_classes)
            state_dict = torch.load(model_path, map_location='cpu')

            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v

            self.model.load_state_dict(new_state_dict)
            self.model = self.model.to(device).float()
            self.model.eval()

            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
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
        model_path="D:/pyproj/GraduateProject/Data/enhance_f_vgg.pth",
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
        # Potato
        'Potato___Early_Blight': '马铃薯早疫病',
        'Potato___Late_Blight': '马铃薯晚疫病',
        # Rice
        'Rice___Brown_Spot': '水稻胡麻斑病',
        'Rice___Leaf_Blast': '水稻稻瘟病',
        'Rice___Neck_Blast': '水稻穗颈瘟',
        # Wheat
        'Wheat___Brown_Rust': '小麦杆锈病',
        'Wheat___Yellow_Rust': '小麦条锈病',
        # Sugarcane
        'sugarcane___Bacterial Blight': '甘蔗白条病',
        'sugarcane___Red Rot': '甘蔗赤腐病',
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
                except Exception as e:
                    print(f"作物识别错误: {str(e)}")

            # 2. YOLO病害检测
            disease_data = []
            if yolo_model and crop_class != "未知作物":
                try:
                    # 动态参数调整
                    conf_threshold = 0.3 if "玉米" in crop_class else 0.25
                    iou_threshold = 0.5 if "马铃薯" in crop_class else 0.45

                    yolo_results = yolo_model.predict(
                        np.array(image),
                        imgsz=320,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        device=device,
                        verbose=False,
                        agnostic_nms=True
                    )

                    # 处理并过滤结果
                    raw_results = CropsService.process_yolo_results(yolo_results)
                    filtered = [res for res in raw_results
                                if CropsService.is_related_disease(crop_class, res['class_name'])]

                    # 转换为中文并查询数据库
                    for disease in filtered:
                        en_name = disease['class_name'].replace(' ', '_')  # 统一为下划线格式
                        zh_name = CropsService.YOLO_DISEASE_MAPPING.get(
                            disease['class_name'],  # 原始名称查询
                            CropsService.YOLO_DISEASE_MAPPING.get(
                                en_name,  # 下划线格式查询
                                disease['class_name']))  # 如果都没有匹配到，返回原始名称

                        # 数据库查询
                        record = Crops.query.filter(
                            Crops.Cclass == crop_class,
                            Crops.Cdisaster == zh_name  # 改为精确匹配
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
            '玉米': ['Common_Rust', 'Gray_Leaf_Spot', 'Northern_Leaf_Blight'],
            '马铃薯': ['Early_Blight', 'Late_Blight'],
            '水稻': ['Brown_Spot', 'Leaf_Blast', 'Neck_Blast'],
            '小麦': ['Brown_Rust', 'Yellow_Rust'],
            '甘蔗': ['Bacterial Blight', 'Red Rot']
        }
        return any(
            keyword in disease_name
            for keyword in disease_mapping.get(crop_type, [])
        )