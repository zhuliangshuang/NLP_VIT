# -*- coding:utf-8 -*-
# Author: 朱良双
# Date: 2022-12-20
import torch
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from VisionTransformer_model import VIT  # 导入我们之前定义的 VIT B-16 模型
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 绘图显示中文
# 参数设置
batch_size = 12  # 每个step处理12张图片
epochs = 80  # 训练10轮
best_loss = 1  # 当验证集损失小于1时才保存权重
best_acc = 0

# 数据集目录位置
filepath = './flower_photos/'

# 预训练权重位置官网所下载的与训练权重
weightpath = './pre_weights/vit_base_patch16_224.pth'
#weightpath = './pre_weights/vit_base_patch16_224.pth'
# 训练时保存权重文件的位置
savepath = './flower_weights/'


# 获取GPU设备，检测到了就用GPU，检测不到就用CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# 数据集处理
# 定义预处理方法
data_transform = {
    # 训练集预处理方法
    'train': transforms.Compose([
        transforms.Resize((224, 224)),  # 将原始图片缩放至224*224大小
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # numpy类型变tensor，维度调整，数据归一化
        transforms.Normalize(mean=[0.46596512, 0.42487496, 0.30391714],  std=[0.2452963, 0.21879682, 0.22424108])  # 对图像的三个通道分别做标准化 原本默认【0.5 0.5 0.5】
    ]),

    # 验证集预处理方法
    'val': transforms.Compose([
        transforms.Resize((224, 224)),  # 将输入图像缩放至224*224大小
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.46596512, 0.42487496, 0.30391714],  std=[0.2452963, 0.21879682, 0.22424108])
    ])
}

# 加载数据集
datasets = {
    'train': datasets.ImageFolder(filepath + 'train', transform=data_transform['train']),  # 读取训练集
    'val': datasets.ImageFolder(filepath + 'val', transform=data_transform['val'])  # 读取验证集
}

# 构造数据集
dataloader = {
    'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=True),  # 构造训练集
    'val': DataLoader(datasets['val'], batch_size=batch_size, shuffle=True)  # 构造验证集
}


# 查看数据集信息
train_num = len(datasets['train'])  # 查看训练集数量
val_num = len(datasets['val'])  # 查看验证集数量

print("训练集数量",train_num)
print("验证集数量",val_num)
# 查看分类类别及其索引
class_names = dict((v, k) for k, v in datasets['train'].class_to_idx.items())
print("分别种类：",class_names)

# 从训练集中取出一个batch，接收图片及其标签
train_imgs, train_labels = next(iter(dataloader['train']))
print('img:', train_imgs.shape, 'labels:', train_labels.shape)


# 数据可视化

# 从数据集中取出10张图及其标签
frames = train_imgs[:12]
frames_labels = train_labels[:12]

# 将数据类型从tensor变回numpy
frames = frames.numpy()

# 维度调整 [b,c,h,w]==>[b,h,w,c]
frames = np.transpose(frames, [0, 2, 3, 1])

# 对图像做反标准化处理
mean = [0.46596512, 0.42487496, 0.30391714]  # 均值
std = [0.2452963, 0.21879682, 0.22424108]   # 标准化
# 图像的每个通道的特征图乘标准化加均值
frames = frames * std + mean

# 将像素值限制在0-1之间
frames = np.clip(frames, 0, 1)

# 绘制12张图像及其标签
# plt.figure()  # 创建画板
# for i in range(12):
#     plt.subplot(3, 4, i + 1)
#     plt.imshow(frames[i])
#     plt.axis('off')  # 不显示坐标刻度
#     plt.title(class_names[frames_labels[i].item()])  # 显示每张图片的标签
# plt.tight_layout()  # 轻量化布局
# plt.show()


# 模型加载，迁移学习

# 接收VIT模型，5分类
model = VIT(num_classes=5)

# 加载预训练权重文件，文件中的分类层神经元个数是1k
pre_weights = torch.load(weightpath, map_location=device)

# 删除权重文件中不需要的层，保留除了分类层以外的所有层的权重
del_keys = ['head.weight', 'head.bias']

# 删除字典中的对应key
for k in del_keys:
    del pre_weights[k]

# 将修改后的权重加载到模型上
# 当strict=True,要求预训练权重层数的键值与新构建的模型中的权重层数名称完全吻合
missing_keys, unexpected_keys = model.load_state_dict(pre_weights, strict=False)
print('miss:', len(missing_keys), 'unexpected:', len(unexpected_keys))

# model.parameters() 代表网络的所有参数
for params in model.parameters():
    params.requires_grad = True  # 所有权重参与训练可以更新


# 网络编译
# 将模型搬运至GPU上
model.to(device)
# 定义交叉熵损失
loss_function = nn.CrossEntropyLoss()

# 获取所有需要梯度更新的权重参数
params_optim = []
# 遍历网络的所有权重
for p in model.parameters():
    if p.requires_grad is True:  # 查看权重是否需要更新
        params_optim.append(p)  # 保存所有需要更新的权重

print('训练参数 长度：', len(params_optim))

# 定义优化器，定义学习率，动量，正则化系数
optimizer = optim.SGD(params_optim, lr=0.001, momentum=0.9, weight_decay=3e-4)

# 训练阶段

for epoch in range(epochs):

    print('=' * 30)  # 显示当前是第几个epoch

    # 将模型设置为训练模式
    model.train()
    # 记录一个epoch的训练集总损失
    total_loss = 0.0

    # 每个step训练一个batch，每次取出一个数据集及其标签
    for step, (images, labels) in enumerate(dataloader['train']):

        # 将数据集搬运到GPU上
        images, labels = images.to(device), labels.to(device)
        # 梯度清零，因为梯度是累加的
        optimizer.zero_grad()
        # 前向传播==>[b,5]
        logits = model(images)  # 得到每张图属于5个类别的分数
        #print(logits)

        # （1）损失计算
        # 计算每个step的预测值和真实值的交叉熵损失
        loss = loss_function(logits, labels)
        # 累加每个step的损失
        total_loss += loss

        # （2）反向传播
        # 梯度计算
        with torch.no_grad():  # 接下来不计算梯度
            loss.backward()
        # 梯度更新
            optimizer.step()

        # 每89个epoch打印一次损失值
        if step % 89 == 0:
            print(f'step:{step}, train_loss:{loss}')

    # 计算一个epoch的训练集平均损失
    train_loss = total_loss / len(dataloader['train'])

    # --------------------------------------------- #
    # 验证训练
    # --------------------------------------------- #
    model.eval()  # 切换到验证模式

    total_val_loss = 0.0  # 记录一个epoch的验证集总损失
    total_val_correct = 0  # 记录一个epoch中验证集一共预测对了几个

    with torch.no_grad():  # 接下来不计算梯度
        # 每个step验证一个batch
        for (images, labels) in dataloader['val']:
            # 将数据集搬运到GPU上
            images, labels = images.to(device), labels.to(device)
            # 前向传播[b,c,h,w]==>[b,5]
            logits = model(images)

            # 计算损失
            # 计算每个batch的预测值和真实值的交叉熵损失
            loss = loss_function(logits, labels)

            # 累加每个batch的损失，得到一个epoch的总损失
            total_val_loss += loss

            # 计算准确率
            # 找到预测值对应的最大索引，即该图片对应的类别
            pred = logits.argmax(dim=1)  # [b,3]==>[b]
            # 比较预测值和标签值，计算每个batch有多少预测对了
            val_correct = torch.eq(pred, labels).float().sum()

            # 累加每个batch的正确个数，计算整个epoch的正确个数
            total_val_correct += val_correct

        # 计算一个epoch的验证集的平均损失和平均准确率
        val_loss = total_val_loss / len(dataloader['val'])
        val_acc = total_val_correct / val_num

        # 打印每个epoch的训练集平均损失，验证集平均损失和平均准确率
        print('-' * 30)
        print(f'train_loss:{train_loss}, val_loss:{val_loss}, val_acc:{val_acc}')

    #保存权重
        file = open('./flower_weights/12_size_acc.txt', 'a')
        file.write(' ' + str(val_acc.item()))
        file.close()
        file = open('./flower_weights/12_size_tranloss.txt', 'a')
        file.write(' ' + str(val_loss.item()))
        file.close()
        # 保存最小损失值对应的权重文件
        if val_loss < best_loss:
        #if val_acc > best_acc:
            # 权重文件名称
            savename = savepath + f'12_size_epoch{epoch}_valacc{round(val_acc.item() * 100)}%_' + 'VIT.pth'
            # 保存该轮次的权重
            torch.save(model.state_dict(), savename)
            # 切换最小损失值
            best_loss = val_loss
            best_acc = val_acc
            # 打印结果
            print(f'weights has been saved, best_loss has changed to {val_loss}')