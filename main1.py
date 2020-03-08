# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: user
"""

import argparse
import torch
from flyai.dataset import Dataset
from model import Model
from net import Net
import numpy as np
from path import MODEL_PATH
from torchvision.models import *
from torch import nn,optim
from flyai.utils.log_helper import train_log
from config import params
from radam import RAdam
from efficientnet_pytorch import EfficientNet


'''
样例代码仅供参考学习，可以自己修改实现逻辑。
Tensorflow模版项目下载： https://www.flyai.com/python/tensorflow_template.zip
PyTorch模版项目下载： https://www.flyai.com/python/pytorch_template.zip
Keras模版项目下载： https://www.flyai.com/python/keras_template.zip
第一次使用请看项目中的：第一次使用请读我.html文件
常见问题请访问：https://www.flyai.com/question
意见和问题反馈有红包哦！添加客服微信：flyaixzs
'''

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def cal_acc(y_, y):
    n = y.size()[0]
    y_pred = torch.argmax(y_, axis=1)
    true = torch.sum(y == y_pred).float()
    acc = true/n
#    print(true, n, acc)
    return acc

def train(net, x, y, optimizer, criterion):
    net.train()
    x = torch.tensor(x.transpose((0, 3, 1, 2))).to(device)
    y = torch.tensor(np.argmax(y, axis=1)).to(device)#.unsqueeze(-1).float()
    y_ = net(x)

    loss = criterion(y_, y)
    acc = cal_acc(y_, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), acc.item()

def valid(net, x, y, optimizer, criterion):
    net.eval()
    with torch.no_grad():
        x = torch.tensor(x.transpose((0, 3, 1, 2))).to(device)
        y = torch.tensor(np.argmax(y, axis=1)).to(device)#.unsqueeze(-1).float()
        y_ = net(x)
        acc = cal_acc(y_, y)
        loss = criterion(y_, y)
    return loss.item(), acc.item()
'''
项目的超参
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=1, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=1, type=int, help="batch size")
args = parser.parse_args()
max_epoch = args.EPOCHS
'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
dataset.image_aug((448, 448), horizontal_flip=True, rotation_range=15,zca_whitening=True, vertical_flip=False)#,seed=6
model = Model(dataset)

'''
实现自己的网络机构
'''
# 判断gpu是否可用
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)
#net = Net().to(device)
#net  = densenet201(pretrained=True)#加载已经训练好的模型
#net = EfficientNet.from_pretrained('efficientnet-b0')
# net = resnet152(pretrained=True)
#net = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl')
# print(net)
# raise RuntimeError
# num_ftrs = net.classifier.in_features
# net.classifier = nn.Linear(num_ftrs, 4)

# num_ftrs = net.fc.in_features
# net.fc = nn.Linear(num_ftrs, 4)
#
# net = net.to(device)
total_step = dataset.get_step()
# optimizer = optim.Adam(net.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
#optimizer = RAdam(net.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
# schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.333, patience=0.1*total_step, verbose=True)
# criterion = nn.CrossEntropyLoss()

'''
dataset.get_step() 获取数据的总迭代次数

'''
best_score = 0
min_loss = 1000
print(total_step)
print('------------------start training------------------------')
# train_losses = AverageMeter()
# valid_losses = AverageMeter()
# train_accs = AverageMeter()
# valid_accs = AverageMeter()
for step in range(total_step):
    x_train, y_train = dataset.next_train_batch()
    # x_val, y_val = dataset.next_validation_batch()

    # print(x_train.shape, x_val.shape)
    # raise RuntimeError

    # train_loss, train_acc = train(net, x_train, y_train, optimizer,criterion)
    # valid_loss, valid_acc = valid(net, x_val, y_val, optimizer,criterion)
    # train_losses.update(train_loss)
    # valid_losses.update(valid_loss)
    # train_accs.update(train_acc)
    # valid_accs.update(valid_acc)
    # final_loss = valid_losses.avg
    # schedule.step(final_loss)
    # '''
    # 实现自己的模型保存逻辑
    # '''
    #
    # if valid_loss < min_loss:
    #     model.save_model(net, MODEL_PATH, overwrite=True)
    #     min_loss = valid_loss
    # print('-----------------------------------------')
    print(str(step + 1) + "/" + str(dataset.get_step()))
    # print('train loss:%0.4f'%train_losses.avg, 'train acc:%0.2f'%(train_accs.avg*100)+'%', 'lr:', optimizer.param_groups[0]['lr'])
    # print('valid loss:%0.4f'%valid_losses.avg, 'valid acc:%0.2f'%(valid_accs.avg*100)+'%')
    # train_log(train_loss=train_losses.avg, train_acc=train_accs.avg, val_loss=valid_losses.avg, val_acc=valid_accs.avg)