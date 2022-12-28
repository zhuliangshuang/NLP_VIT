# -*- coding:utf-8 -*-
# Author: 朱良双
# Date: 2022-12-20
import torchvision.datasets as datasets
import torch
from torchvision import transforms


def getStat(train_data):
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for i in range(3):
            mean[i] += X[:, i, :, :].mean()
            std[i] += X[:, i, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())

if __name__ == '__main__':
    tranform = transforms.Compose([
        transforms.Resize((224, 224)),  # 将原始图片缩放至224*224大小
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor()
    ])
    train_dataset = datasets.ImageFolder(root=r'./flower_photos/train', transform=tranform)
    print(getStat(train_dataset))

