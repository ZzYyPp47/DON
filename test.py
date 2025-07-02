# -*- coding:utf-8 -*-
'''
@Author: ZYP
@contact: 3137168510@qq.com
@Time: 2024/2/11 21:59
@version: 1.0
@File: test.py
'''

import torch
import torch.nn as nn
import matplotlib as plt
import random
from DON import *
from data import *
from NN.FNN import FNN
from NN.ResFNN import ResFNN
from NN.Dropout_FNN import Dropout_FNN
from scipy.io import loadmat
from torch.utils.data import DataLoader


def test():
    # 初始化
    seed = 0 # 随机种子
    p = 128 # p的维数
    Arc_branch = [100, 128, 128, p] # 神经网络架构
    Arc_trunk = [1, 128, p]  # 神经网络架构
    func = nn.Tanh() # 确定激活函数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 自动选择可用设备,优先GPU
    learning_rate = 0.0001
    name = 'DON'
    init_method = torch.nn.init.kaiming_uniform_ # 设置神经网络参数初始化方法
    total_epochs = 1000
    batch_size = 64
    dropout_rate = 0.1
    loss_func = nn.MSELoss().to(device) # 确定损失计算函数
    loss_weight = [1,0] # loss各项权重(pde,bound,ini,real)
    mini_batch_flag = False # 影响数据集导入的过程

    # 建立模型
    set_seed(seed)  # 设置确定的随机种子
    branch = FNN(Arc_branch,func,device)
    trunk = FNN(Arc_trunk,func,device)
    point = create_point(device,mini_batch_flag,Arc_branch[0],Arc_trunk[0])
    DON_demo = DON(name, [total_epochs,batch_size], loss_func, [branch, trunk], point, learning_rate, init_method, loss_weight, device)
    # 务必确定model与point位于同一设备!!

    # 训练
    start_time = time.time()
    DON_demo.train_all()
    end_time = time.time()
    print('training terminated,using times:{}s'.format(end_time - start_time))
    DON_demo.save()

    # 画图
    draw(DON_demo,DON_demo.path, device)


# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    np.random.seed(seed)  # 为np设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
        torch.backends.cudnn.deterministic = True  # 使用确定性算法
        torch.backends.cudnn.benchmark = True  # cudnn基准(使用卷积时可能影响结果)

# 画图辅助函数
def draw(DON, load_path, device):
    checkpoint = torch.load(load_path)  # 加载模型
    print('loading from {}'.format(load_path))
    DON.model_list[0].load_state_dict(checkpoint['model_branch'])
    DON.model_list[1].load_state_dict(checkpoint['model_trunk'])
    DON.Epochs_loss = np.array(checkpoint['Epochs_loss'])
    DON.model_list[0].eval()  # 启用评估模式
    DON.model_list[1].eval()  # 启用评估模式
    with torch.no_grad():
        pred = DON.call(DON.loss_computer.point.val_func_sample,DON.loss_computer.point.val_y)
        true = DON.loss_computer.point.val_G_fy
        x = torch.linspace(-0.5,0.5,100).reshape(-1,1)

        relative_l2 = torch.norm(pred - true) / torch.norm(true)
        print('relative_l2 = {}'.format(relative_l2.item()))

        fun_name = ['sin(x)','sec(x)','1.5^x','x^3','x^4','e^x']
        G_fun_name = ['cos(x)','sec(x)tan(x)','1.5^x * ln(1.5)','3x^2','4x^3','e^x']

        for ii in range(len(fun_name)):
            plt.figure()
            l1, = plt.plot(x.cpu(), true[100 * ii:100 * (ii + 1)].cpu(),'b-')
            l2, = plt.plot(x.cpu(), pred[100 * ii:100 * (ii + 1)].cpu(),'r--')
            plt.legend(handles=[l1,l2], labels=['real','pred'], loc='best')
            plt.title(f'f = {fun_name[ii]}, G(f) = {G_fun_name[ii]}')
            plt.grid(axis='both',linestyle='--')
            plt.xlabel("x")
            plt.ylabel("y")

        plt.figure()
        ax1 = plt.gca()
        ax1.semilogy(DON.Epochs_loss[:, 0], DON.Epochs_loss[:, 1], label='Training Loss')
        plt.grid(axis='both', linestyle='--')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Losses with Epochs')

        ax2 = ax1.twinx()
        ax2.semilogy(DON.Epochs_loss[:, 0], DON.Epochs_loss[:, 2], 'r', label='Validation Loss')
        ax2.set_ylabel('Validation Loss')

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()

        handles = handles1 + handles2
        labels = labels1 + labels2
        ax2.legend(handles, labels, loc='best')
        plt.show()





if __name__ == '__main__':
    test()
