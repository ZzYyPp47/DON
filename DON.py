#-*- coding:utf-8 -*-
'''
@Author: ZYP
@contact: 3137168510@qq.com
@Time: 2024/1/25 15:26
@version: 1.0
@File: pinn.py
'''

# 导入必要的包
import torch
import time
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import deque
from torch.utils.data import DataLoader
import numpy as np
import datetime
from math import ceil
from warnings import warn
from loss import *


class DON(object):
    # 初始化
    def __init__(self,name,training_info,loss_func,model_list,point,learning_rate,weights_init,loss_weight,device):
        self.total_epochs, self.batch_size = training_info # 训练所需要的信息
        self.loss_computer = LossCompute(self.call,loss_func, point, loss_weight)  # 损失计算(接口)
        self.path = 'save/'+name+'.pth'# '_'+datetime.date.today().strftime('%Y-%m-%d,%H:%M:%S')+'.pth'
        self.device = device # 使用的设备
        self.Epochs_loss = [] # 记录loss
        self.stop_que = deque([]) # 用以进行移动平均
        self.step, self.patient_step, self.count, self.best = 300, 500, 0, float('inf')  # 移动窗口大小、容忍大小、容忍计数、最佳模型
        self.weights_init = weights_init # 确定神经网络初始化的方式
        self.loss_func = loss_func # 损失函数(接口)
        self.model_list = model_list # 神经网络表(接口)
        self.bias = torch.tensor(0, dtype=torch.float32, device=device) # 偏置
        self.opt = torch.optim.Adam(params = [param for model in model_list for param in model.parameters()] + [self.bias],lr = learning_rate,weight_decay=1e-5) # 优化器
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer = self.opt,step_size = 50, gamma = 0.5)
        self.print_model() # 打印model架构

    # 参数初始化
    def weights_initer(self,model):
            if isinstance(model,nn.Conv2d):
                self.weights_init(model.weight.data)
                model.bias.data.zero_()
            elif isinstance(model,nn.Linear):
                self.weights_init(model.weight.data)
                model.bias.data.zero_()

    # 训练模型
    def train_all(self):
        self.model_list[0].train()  # 启用训练模式
        self.model_list[1].train()  # 启用训练模式
        print('start training,using seed:{}'.format(torch.initial_seed()))
        for epoch in range(self.total_epochs):
            batchset = DataLoader(self.loss_computer.point, batch_size = self.batch_size, shuffle = True, drop_last = True)
            for idx, batch_data in enumerate(batchset):
                self.opt.zero_grad() # 清零梯度信息
                output = self.call(batch_data['func_sample'], batch_data['y'])
                true = batch_data['G_fy']
                Loss = self.loss_computer.loss(output,true)
                Loss.backward() # 反向计算出梯度
                # val_loss = self.val_loss().item()
                # self.Epochs_loss.append([epoch + 1, Loss.item(), val_loss])
                # flag = self.stop_check(val_loss, self.step, self.patient_step)
                # if flag:
                #     break
                self.opt.step() # 更新参数
                # self.scheduler.step() # 学习率退火
                # if (epoch + 1) % 500 == 0:
                #     print(f'Epoch:{epoch + 1}/{self.total_epochs},Loss={Loss.item()},Val_loss={val_loss}')
                if (idx + 1) % 500 == 0:
                    val_loss = self.val_loss().item()
                    print(f'Epoch:{epoch + 1}/{self.total_epochs},Batch:{idx + 1}/{self.loss_computer.point.len // self.batch_size},Loss={Loss.item()},Val_loss={val_loss}')# Batch:{idx + 1}/{self.loss_computer.point.len // self.batch_size}
            val_loss = self.val_loss().item()
            self.Epochs_loss.append([epoch + 1, Loss.item(), val_loss])
            flag = self.stop_check(val_loss, self.step, self.patient_step)
            if flag:
                break
            self.scheduler.step() # 学习率退火
            torch.cuda.empty_cache()

        # 启用LBFGS优化
        print('now using LBFGS...')
        self.opt = torch.optim.LBFGS(params = [param for model in self.model_list for param in model.parameters()] + [self.bias], history_size=100, tolerance_change=0, tolerance_grad=1e-08,max_iter=5000, max_eval=12000)
        self.count = 0 # 先清零之前计数
        try:
            self.opt.step(self.closure)
        except EarlyExit as e:
            pass





     # 保存模型参数
    def save(self):
        state_dict = {"model_branch": self.model_list[0].state_dict(),"model_trunk": self.model_list[1].state_dict(),"Epochs_loss":self.Epochs_loss}
        torch.save(state_dict,self.path)
        print('model saved to {}'.format(self.path))

    # 验证函数
    def val_loss(self):
        output = self.call(self.loss_computer.point.val_func_sample, self.loss_computer.point.val_y)
        true = self.loss_computer.point.val_G_fy
        return self.loss_computer.loss_operator(output,true)

    # 调用函数
    def call(self,branch_input,trunk_input):
        branch_output = self.model_list[0](branch_input)
        trunk_output = self.model_list[1](trunk_input)
        return torch.sum(branch_output * trunk_output,dim=(1, ), keepdim=True) + self.bias

    # 输入检查
    def check(self):
        if isinstance(self.batch_size,float):
            if self.batch_size <= 0:
                raise Exception('所输入的 batch_size 不受支持,batch_size 的值为: {}'.format(self.batch_size))
            else:
                self.batch_size = int(self.batch_size * self.loss_computer.point.len)
        elif isinstance(self.batch_size,int):
            if self.batch_size <= 0:
                raise Exception('所输入的 batch_size 不受支持,batch_size 的值为: {}'.format(self.batch_size))
        else:
            raise Exception('所输入的 batch_size 不受支持,batch_size 的值为: {}'.format(self.batch_size))
        if self.batch_size > self.loss_computer.point.len:
            warn('输入值{}大于训练集长度{},将进行循环索引.'.format(self.batch_size, self.loss_computer.point.len))

    # 打印架构
    def print_model(self):
        for ii in range(len(self.model_list)):
            self.model_list[ii] = self.model_list[ii].to(self.device)
            self.model_list[ii].model.apply(self.weights_initer)  # 神经网络初始化
            print(self.model_list[ii])

    # 早停法+瓶颈停止(step步移动平均)、与最佳模型保存
    def stop_check(self,loss,step,patient_step):
        improvement_threshold = 0.99  # 只有当验证损失改善超过1%时才认定有“突破”
        if loss < self.best:
            self.best = loss
            self.count = 0  # 重置计数
            self.save()
            print(f'find best model at epoch = {self.Epochs_loss[-1][0]}, val_loss = {loss}')
        if len(self.stop_que) < step:
            # 元素不足则加入,无条件继续运行
            self.stop_que.append(loss)
            return False
        else:
            avg = sum(self.stop_que) / len(self.stop_que) # 计算均值
            if (loss >= avg):
                # 超过窗口平均值，进入容忍期
                self.count += 1
            elif loss <= (min(self.stop_que) * improvement_threshold):
                self.count = 0 # 计数器重置
            # 删除左元素、加入新元素
            self.stop_que.popleft()
            self.stop_que.append(loss)
        if self.count >= patient_step:
            print(f'reach patient step. (avg = {avg}, val_loss = {loss})')
            return True

    # 为LBFGS优化器准备的闭包函数
    def closure(self):
        self.opt.zero_grad()  # 清零梯度信息
        output = self.call(self.loss_computer.point.func_sample, self.loss_computer.point.y)
        true = self.loss_computer.point.G_fy
        Loss = self.loss_computer.loss(output, true)
        Loss.backward()  # 反向计算出梯度
        val_loss = self.val_loss().item()
        flag = self.stop_check(val_loss, self.step, self.patient_step)
        if flag:
            raise EarlyExit
        self.Epochs_loss.append([self.Epochs_loss[-1][0] + 1, Loss.item(), val_loss])
        print(f'loss:{Loss.item()},val_loss:{val_loss}')
        return Loss.item()

class EarlyExit(Exception):
    # 自定义异常用于提前退出闭包函数
    pass