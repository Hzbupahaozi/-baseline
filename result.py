# test_images文件夹批量预测
import os
import uuid
import csv
from PIL import Image
import torch
import torchvision.transforms as transforms

# 定义需要预测的类别
classes = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9']
# 定义图像变换
pre_train_weights = './checkpoint/resnet50_apple/model.pth'
model = torch.load(pre_train_weights)
model.eval()
transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor,
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# 定义预测函数
def predict_image(image_path):
    # 打开图像并进行变换
    image = Image.open(image_path)
    image_tensor = transforms(image).unsqueeze(0)

    # 使用模型进行预测
    output = model(image_tensor)
    _, predicted = torch.max(output, 1)
    # 返回预测结果
    return classes[predicted]


# 定义文件夹路径
folder_path = './data/test'
# 获取所有图像的文件名
image_names = os.listdir(folder_path)

# 定义csv文件的列名和路径
csv_columns = ['uuid', 'label']
csv_file = './result.csv'

# 创建csv文件并写入列名
with open(csv_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=csv_columns)
    writer.writeheader()

# 遍历所有图像进行预测，并将预测结果写入csv文件
for image_name in image_names:
    # 获取图像文件路径
    image_path = os.path.join(folder_path, image_name)

    # 进行预测
    predicted_class = predict_image(image_path)

    # 将预测结果写入csv文件
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writerow({'uuid': str(uuid.uuid4()), 'label': predicted_class})
