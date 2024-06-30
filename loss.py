# -*- coding:utf-8 -*-
'''
@Author: ZYP
@contact: 3137168510@qq.com
@Time: 2024/2/12 12:41
@version: 1.0
@File: loss.py
'''

import torch



class LossCompute:
    def __init__(self,call,loss_func, point, loss_weight):
        self.call = call # 回调函数
        self.loss_func = loss_func
        self.point = point
        self.loss_weight = loss_weight # 损失函数各项的权重分配

    # 计算梯度
    def gradient(self,func,var,order = 1):
        if order == 1:
            return torch.autograd.grad(func,var,grad_outputs=torch.ones_like(func),create_graph=True,only_inputs=True)[0]
        else:
            out = self.gradient(func,var)# 不要加order(以正常计算1阶导),否则会无限循环调用！
            return self.gradient(out,var,order - 1)

    def loss_operator(self,output,true):
        return self.loss_func(output, true)

    def loss_physics(self,output,x):
        d_x = self.gradient(output,x)
        return self.loss_func(d_x, torch.zeros_like(d_x))

    def loss(self,output,true,x = 0):
        if self.loss_weight[1] != 0:
            return self.loss_weight[0] * self.loss_operator(output,true) + self.loss_weight[1] * self.loss_physics(output,x)
        else:
            return self.loss_weight[0] * self.loss_operator(output,true)
