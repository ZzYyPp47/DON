# -*- coding:utf-8 -*-
'''
@Author: ZYP
@contact: 3137168510@qq.com
@Time: 2024/5/2 12:02
@version: 1.0
@File: Dropout_FNN.py
'''


import torch.nn as nn

class Dropout_FNN(nn.Module):
    def __init__(self, Arc, func, device, input_transform = None, output_transform = None, dropout_rate = 0.5):
        super(Dropout_FNN, self).__init__()  # 调用父类的构造函数
        self.input_transform = input_transform  # 输入特征转换
        self.output_transform = output_transform  # 输出特征转换
        self.func = func  # 定义激活函数
        self.Arc = Arc  # 定义网络架构
        self.dropout_rate = dropout_rate  # 添加Dropout率参数
        self.device = device
        self.model = self.create_model().to(self.device)
        # print(self.model)

    def create_model(self):
        layers = []
        for ii in range(len(self.Arc) - 1):
            layers.append(nn.Linear(self.Arc[ii], self.Arc[ii + 1]))
            if ii < len(self.Arc) - 2:  # 在最后一层前，加入激活函数和Dropout
                layers.append(self.func)
                layers.append(nn.Dropout(self.dropout_rate))  # 在每个非输出层后添加Dropout层
        return nn.Sequential(*layers)  # 利用*将layers列表解包成独立变量传入sequential

    def forward(self, x):
        if self.input_transform is not None:
            x = self.input_transform(x)
        out = self.model(x)
        if self.output_transform is not None:
            out = self.output_transform(out, x)
        return out