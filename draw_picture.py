# -*- coding:utf-8 -*-
# Author: 朱良双
# Date: 2022-12-20
import numpy

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 绘图显示中文

#DeepW_loss = numpy.loadtxt('./flower_weights/Deep_W/tranloss.txt')
VIT_loss = numpy.loadtxt('./flower_weights/12_size_tranloss.txt')
big_sizeDeepW_loss = numpy.loadtxt('./flower_weights/Deep_W/12_size_tranloss.txt')
#DeepW_acc = numpy.loadtxt('./flower_weights/Deep_W/acc.txt')
VIT_acc = numpy.loadtxt('./flower_weights/12_size_acc.txt')
big_sizeDeepW_acc = numpy.loadtxt('./flower_weights/Deep_W/12_size_acc.txt')
plt.figure(figsize=(10, 6), dpi=100)
dt = numpy.arange(1, 81)

#plt.plot(dt,DeepW_acc, c='red', label="Deep_VIT_Acc")
plt.plot(dt,VIT_acc, c='blue', label="VIT_Acc")
plt.plot(dt,big_sizeDeepW_acc, c='red', label="DeepVIT_Acc")

#plt.scatter(dt,DeepW_acc, c='red')
plt.scatter(dt,VIT_acc, c='blue')
plt.scatter(dt,big_sizeDeepW_acc, c='red')
plt.legend(loc='best')
plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel("epochs", fontdict={'size': 16})
plt.ylabel("准确度", fontdict={'size': 16})
plt.title("训练数据", fontdict={'size': 20})
plt.show()


plt.figure(figsize=(10, 6), dpi=100)
dt = numpy.arange(1, 81)

#plt.plot(dt,DeepW_loss, c='red', label="Deep_VIT_loss")
plt.plot(dt,VIT_loss, c='blue', label="VIT_loss")
plt.plot(dt,big_sizeDeepW_loss, c='red', label="DeepVIT_loss")

#plt.scatter(dt,DeepW_loss, c='red')
plt.scatter(dt,VIT_loss, c='blue')
plt.scatter(dt,big_sizeDeepW_loss, c='red')
plt.legend(loc='best')
plt.ylim(0, 5)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel("epochs", fontdict={'size': 16})
plt.ylabel("损失值", fontdict={'size': 16})
plt.title("训练数据", fontdict={'size': 20})
plt.show()

# fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 6), dpi=100)
# dt = numpy.arange(1, 81)
# DeepW_loss = numpy.loadtxt('./flower_weights/Deep_W/tranloss.txt')
# VIT_loss = numpy.loadtxt('./flower_weights/tranloss.txt')
# big_sizeDeepW_loss = numpy.loadtxt('./flower_weights/Deep_W/12_size_tranloss.txt')
# DeepW_acc = numpy.loadtxt('./flower_weights/Deep_W/acc.txt')
# VIT_acc = numpy.loadtxt('./flower_weights/acc.txt')
# big_sizeDeepW_acc = numpy.loadtxt('./flower_weights/Deep_W/12_size_acc.txt')
#
# # my_data_acc = numpy.loadtxt('./flower_weights/acc.txt')
# # my_data_loss = numpy.loadtxt('./flower_weights/tranloss.txt')
# y_data = [DeepW_loss, VIT_loss,big_sizeDeepW_loss,DeepW_acc,VIT_acc,big_sizeDeepW_acc]
# colors = ['red', 'blue','yellow','red', 'blue','yellow']
# line_style = ['-', '-','-', '-','-', '-']
# y_labels = ["Deep_VIT_loss", "VIT_loss","12_sizeDeep_VIT_loss","Deep_VIT_acc", "VIT_acc","12_sizeDeep_VIT_acc"]
# for i in range(6):
#     axs[i].plot(dt, y_data[i], c=colors[i], label=y_labels[i], linestyle=line_style[i])
#     axs[i].scatter(dt, y_data[i], c=colors[i])
#     axs[i].legend(loc='best')
#     if i < 3:
#         axs[i].set_ylim(0.1, 1)
#         #axs[i].set_yticks(range(0.1, 1))
#     else:
#         axs[i].set_ylim(0, 5)
#         #axs[i].set_yticks(range(0, 5))
#     axs[i].grid(True, linestyle='--', alpha=0.5)
#     axs[i].set_xlabel("epochs", fontdict={'size': 16})
#     axs[i].set_ylabel(y_labels[i], fontdict={'size': 16}, rotation=0)
# axs[0].set_title("训练数据loss值", fontdict={'size': 20})
# axs[1].set_title("训练数据acc值", fontdict={'size': 20})
# fig.autofmt_xdate()
# plt.show()

