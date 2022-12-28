# -*- coding:utf-8 -*-
# Author: 朱良双
# Date: 2022-12-20
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
from deep_VIT import VIT
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------- #
# （0）参数设置
# -------------------------------------------------- #
batch_size = 32  # 每次测试32张图

# 测试集文件夹所在位置
file_path = './flower_photos/val'
# 权重参数路径
weights_path = './flower_weights/Deep_W/12_size_epoch78_valacc100%_VIT.pth'#选择训练好的参数最优\

# 获取GPU设备
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# -------------------------------------------------- #
# （1）构造测试集
# -------------------------------------------------- #
# 定义测试集的数据预处理方法
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # 将输入图像的size缩放至224*224
    transforms.ToTensor(),  # numpy边tensor，像素归一化，维度调整
    transforms.Normalize(mean = [0.46596512, 0.42487496, 0.30391714],  std=[0.2452963, 0.21879682, 0.22424108])  # 对每个通道标准化
])

# 加载测试集，并预处理
datasets = datasets.ImageFolder(file_path, transform=data_transforms)

# 构造测试集
dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=True)

# 查看测试集一共有多少张图
test_num = len(datasets)

# 获取测试集的分类类别及其索引
class_names = dict((v, k) for k, v in datasets.class_to_idx.items())


# -------------------------------------------------- #
# （2）绘图展示预测结果
# imgs:代表输入图像[b,c,h,w]，labels代表图像的真实标签[b]
# cls:代表每张图属的类别索引[b]，scores:代表每张图的类别概率[b]
# -------------------------------------------------- #
def im_show(imgs, labels, cls, scores):
    # 从数据集中取出12张图及其标签索引、概率
    frames = imgs[:12]
    true_labels = labels[:12]
    pred_labels = cls[:12]
    pred_scores = scores[:12]

    # 将数据类型从tensor变回numpy
    frames = frames.numpy()
    # 维度调整 [b,c,h,w]==>[b,h,w,c]
    frames = np.transpose(frames, [0, 2, 3, 1])
    # 对图像做反标准化处理
    mean = [0.46596512, 0.42487496, 0.30391714]  # 均值
    std = [0.2452963, 0.21879682, 0.22424108]  # 标准化
    # 图像的每个通道的特征图乘标准化加均值
    frames = frames * std + mean
    # 将像素值限制在0-1之间
    frames = np.clip(frames, 0, 1)

    # 绘制12张图像及其标签
    plt.figure()  # 创建画板
    for i in range(12):
        plt.subplot(3, 4, i + 1)
        plt.imshow(frames[i])
        plt.axis('off')  # 不显示坐标刻度
        # 显示每张图片的真实标签、预测标签、预测概率
        plt.title('true:' + class_names[true_labels[i].item()] + '\n' +
                  'pred:' + class_names[pred_labels[i].item()] + '\n' +
                  'scores:' + str(round(pred_scores[i].item(), 3))
                  )

    plt.tight_layout()  # 轻量化布局
    plt.show()


# -------------------------------------------------- #
# （3）图像预测
# -------------------------------------------------- #
# 模型构建
model = VIT(num_classes=5)
# 加载权重文件
model.load_state_dict(torch.load(weights_path, map_location=device))
# 将模型搬运到GPU上
model.to(device)
# 模型切换成测试模式，切换LN标准化和dropout的工作方式
model.eval()

# 测试阶段不计算梯度
with torch.no_grad():
    # 每次测试一个batch
    for step, (imgs, labels) in enumerate(dataloader):
        # 将数据集搬运到GPU上
        images, labels = imgs.to(device), labels.to(device)
        # 前向传播==>[b,5]
        logits = model(images)
        # 求出图像属于哪个类别索引[b,5]==>[b]
        pred_cls = logits.argmax(dim=1)
        # 计算图像属于每个类别的概率[b,5]==>[b,5]
        predicts = torch.softmax(logits, dim=1)
        # 获取最大预测类别的概率[b,5]==>[b]
        predicts_score, _ = predicts.max(dim=1)

        # 绘制预测结果
        im_show(imgs, labels, pred_cls, predicts_score)