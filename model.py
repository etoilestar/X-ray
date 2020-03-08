# -*- coding: utf-8 -*
import numpy
import os
import torch
from flyai.model.base import Base
from torchvision import transforms
from glob import glob
from path import MODEL_PATH

__import__('net', fromlist=["Net"])

TORCH_MODEL_NAME = "model.pkl"


class Model(Base):
    def __init__(self, data):
        self.data = data
#        self.net_path = os.path.join(MODEL_PATH, TORCH_MODEL_NAME)
        self.net_paths = glob(os.path.join(MODEL_PATH, 'net*.pkl'))#
        self.nets = []
        for i, net_path in enumerate(self.net_paths):
            if os.path.exists(net_path):
                self.nets.append(torch.load(net_path))

    def predict(self, **data):
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        device = torch.device(device)
        if self.net is None:
            self.net = torch.load(self.net_path)
        x_data = self.data.predict_data(**data)
        x_data = x_data.transpose((0, 3, 1, 2))
        x_data = torch.from_numpy(x_data).to(device)

        outputs = self.net(x_data)
        prediction = outputs.cpu().data.numpy()
        prediction = self.data.to_categorys(prediction)
        return prediction

    def predict_all(self, datas):
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        device = torch.device(device)
        # if self.net is None:
        #     self.net = torch.load(self.net_path)
        labels = []
        for data in datas:
            x_data = self.data.predict_data(**data)
            x_data = x_data.transpose((0, 3, 1, 2))
            x_data = torch.from_numpy(x_data).to(device)
            for i, net in enumerate(self.nets):
                if i == 0:
                    outputs = net(x_data)
                else:
                    outputs += net(x_data)
            outputs = outputs / 5
            prediction = outputs.cpu().data.numpy()
            prediction = self.data.to_categorys(prediction)
            labels.append(prediction)
        return labels

    def batch_iter(self, x, y, batch_size=128):
        """生成批次数据"""
        data_len = len(x)
        num_batch = int((data_len - 1) / batch_size) + 1

        indices = numpy.random.permutation(numpy.arange(data_len))
        x_shuffle = x[indices]
        y_shuffle = y[indices]

        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

    def save_model(self, network, path, name=TORCH_MODEL_NAME, overwrite=False):
        super().save_model(network, path, name, overwrite)
        torch.save(network, os.path.join(path, name))