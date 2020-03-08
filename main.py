# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: user
"""

import argparse
import torch
from torch.hub import load_state_dict_from_url
from flyai.dataset import Dataset
from mydata import FlyAIDataset
from model import Model
from tqdm import tqdm
from net import Net
import numpy as np
from path import MODEL_PATH
from torchvision.models import *
from torch import nn,optim
from flyai.utils.log_helper import train_log
from config import params
from radam import RAdam
from efficientnet_pytorch import EfficientNet
from torchtoolbox.tools import mixup_data, mixup_criterion
#from model.classifier import Classifier
from labelsmooth import LabelSmoothSoftmaxCE


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

def train(net, train_loader, optimizer, criterion):
    losses = AverageMeter()
    accs = AverageMeter()
    net.train()
    alpha = 0.1
    for step, (x, y) in enumerate(tqdm(train_loader)):
        x = x.to(device)
        y = y.to(device)#.unsqueeze(-1).float()

        if params['mixup']:
            x, y_a, y_b, lam = mixup_data(x, y, alpha)
            y_ = net(x)
            loss = mixup_criterion(criterion, y_, y_a, y_b, lam)
        else:
            y_ = net(x)
            loss = criterion(y_, y)
        acc = cal_acc(y_, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item())
        accs.update(acc.item())
    return losses.avg, accs.avg

def valid(net, val_loader, optimizer, criterion):
    losses = AverageMeter()
    accs = AverageMeter()
    net.eval()
    with torch.no_grad():
        for step, (x, y) in enumerate(tqdm(val_loader)):
            x = x.to(device)
            y = y.to(device)#.unsqueeze(-1).float()
            y_ = net(x)
            acc = cal_acc(y_, y)
            loss = criterion(y_, y)
            losses.update(loss.item())
            accs.update(acc.item())

    return losses.avg, accs.avg
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
dataset = Dataset()
x_train, y_train, x_val, y_val = dataset.get_all_data() # 获取全量数据
# x_train: [{'image_path': 'img/10479.jpg'}, {'image_path': 'img/14607.jpg'},   {'image_path': 'img/851.jpg'}...]
# y_train: [{'label': 39}, {'label': 4}, {'label': 3}...]
train_dataset = FlyAIDataset(x_train, y_train)
val_dataset = FlyAIDataset(x_val, y_val, train_flag=False)
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.BATCH)
val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=True, batch_size=args.BATCH)

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
nets = []
#net = Net().to(device)
#加载已经训练好的模型
# net = EfficientNet.from_pretrained('efficientnet-b5')
# net = resnet152(pretrained=True)
# net = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl')
# print(net)
# raise RuntimeError
#from flyai.utils import remote_helper
# net  = densenet121(pretrained=True)
# num_ftrs = net1.classifier.in_features
# net1.classifier = nn.Linear(num_ftrs, 4)
# net1 = net1.to(device)
# nets.append(net1)
#efficientnet_pytorch==0.6.1
net1  = densenet121(pretrained=True)
num_ftrs = net1.classifier.in_features
net1.classifier = nn.Linear(num_ftrs, 4)
net1 = net1.to(device)
nets.append(net1)

#
#net = torch.hub.load('moskomule/senet.pytorch','se_resnet50',num_classes=4, pretrained=True, )
#net2 = EfficientNet.from_pretrained('efficientnet-b5')
# 必须使用该方法下载模型，然后加载
#path = remote_helper.get_remote_date('https://www.flyai.com/m/efficientnet_b3_ra-a5e2fbc7.pth')
#net2.load_state_dict(torch.load(path))
#num_ftrs = net2._fc.in_features
#net2._fc = nn.Linear(num_ftrs, 4)
#net2 = net2.to(device)
#nets.append(net2)

# net3 = googlenet(pretrained=True)
# num_ftrs = net3.fc.in_features
# net3.fc = nn.Linear(num_ftrs, 4)
# net3 = net3.to(device)
# nets.append(net3)
#
net4 = wide_resnet101_2(pretrained=True)
num_ftrs = net4.fc.in_features
net4.fc = nn.Linear(num_ftrs, 4)
net4 = net4.to(device)
nets.append(net4)

net5 = resnext101_32x8d(pretrained=True)
num_ftrs = net5.fc.in_features
net5.fc = nn.Linear(num_ftrs, 4)
net5 = net5.to(device)
nets.append(net5)
# optimizer = optim.Adam(net.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
# optimizer = RAdam(net.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
# schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.333, patience=2, verbose=True)
criterion = nn.CrossEntropyLoss()
# criterion = LabelSmoothSoftmaxCE(lb_pos=0.9, lb_neg=5e-3)

'''
dataset.get_step() 获取数据的总迭代次数

'''
best_score = 0

print(max_epoch)
print('------------------start training------------------------')
for i, net in enumerate(nets):
    min_loss = 1000
    print('----------------------start net{}---------------------'.format(i))
    optimizer = RAdam(net.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.333, patience=5, verbose=True)
    for step in range(max_epoch):
    # model.save_model(net, MODEL_PATH, overwrite=True)
    # break
        train_loss, train_acc = train(net, train_loader, optimizer,criterion)
        valid_loss, valid_acc = valid(net, val_loader, optimizer,criterion)
        schedule.step(valid_loss)
    # '''
    # 实现自己的模型保存逻辑
    # '''
    #


        if valid_loss < min_loss:
            model.save_model(net, MODEL_PATH, name='net_'+str(i)+'.pkl', overwrite=False)
            min_loss = valid_loss
        print('-----------------------------------------')
        print('epoch:', str(step + 1) + "/" + str(max_epoch))
        print('train loss:%0.4f'%train_loss, 'train acc:%0.2f'%(train_acc*100)+'%', 'lr:', optimizer.param_groups[0]['lr'])
        print('valid loss:%0.4f'%valid_loss, 'valid acc:%0.2f'%(valid_acc*100)+'%')
        train_log(train_loss=train_loss, train_acc=train_acc, val_loss=valid_loss, val_acc=valid_acc)
        if optimizer.param_groups[0]['lr'] < 0.1*params['lr']:
            break