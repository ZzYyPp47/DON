# -*- coding:utf-8 -*-
'''
@Author: ZYP
@contact: 3137168510@qq.com
@Time: 2024/2/11 0:21
@version: 1.0
@File: data.py
'''

import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
import numpy as np
import h5py

class create_point(Dataset):
    def __init__(self,device,mini_batch_flag,b_size,t_size):
        super(create_point, self).__init__()  # 调用父类的构造函数
        self.device = device
        self.b_size = b_size
        self.t_size = t_size
        self.mini_batch_flag = mini_batch_flag
        self.data()
        self.len = len(self.func_sample)

    def data(self):
        file_d = 'data.mat'
        data = np.transpose(h5py.File(file_d)['data'])
        if self.mini_batch_flag == False:
            self.func_sample = torch.tensor(data[:, 0:self.b_size],dtype=torch.float32,device=self.device)
            self.y = torch.tensor(data[:, self.b_size:self.b_size + self.t_size], dtype=torch.float32, device=self.device)
            self.G_fy = torch.tensor(data[:,self.b_size + self.t_size:self.b_size + self.t_size + 1], dtype=torch.float32, device=self.device)
        else:
            self.func_sample = torch.tensor(data[:, 0:self.b_size],dtype=torch.float32,device='cpu')
            self.y = torch.tensor(data[:, self.b_size:self.b_size + self.t_size], dtype=torch.float32, device='cpu')
            self.G_fy = torch.tensor(data[:,self.b_size + self.t_size:self.b_size + self.t_size + 1], dtype=torch.float32, device='cpu')

        file_v = 'test.mat'
        val_data = np.transpose(h5py.File(file_v)['test'])
        self.val_func_sample = torch.tensor(val_data[:, 0:self.b_size],dtype=torch.float32,device=self.device)
        self.val_y = torch.tensor(val_data[:, self.b_size:self.b_size + self.t_size], dtype=torch.float32, device=self.device)
        self.val_G_fy = torch.tensor(val_data[:,self.b_size + self.t_size:self.b_size + self.t_size + 1], dtype=torch.float32, device=self.device)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # 为每类获取 index，此处采用循环索引方式以循环采样每种类型的点
        idx_func_sample = idx % self.len
        idx_y = idx % self.len
        idx_G_fy = idx % self.len

        return {
            'func_sample': self.func_sample[idx_func_sample],
            'y': self.y[idx_y],
            'G_fy': self.G_fy[idx_G_fy],
        }
