import torch
import torch.nn as nn
import numpy as np
import random
from BKAN import KANLayer

# Define the combined model  
# class KANModel(nn.Module):
#     def __init__(self):
#         super(KANModel, self).__init__()
#         self.kan_layer1 = KANLayer(2 ,5)    
#         # self.ln1 = nn.Tanh()
#         # self.ln1 = nn.Sigmoid()
#         self.kan_layer2 = KANLayer(5, 2)  

#     def forward(self, x):
#         x = x.view(-1, 2)  # Flatten the input tensor
#         x = self.kan_layer1(x)
#         x = self.kan_layer2(x)
#         return x
    
# torch.set_default_dtype(torch.float64) # 设置默认张量数据类型为双精度浮点数

class KANModel(nn.Module):
    def __init__(self, width=None,  grid_range=None ,seed=0):
        super(KANModel, self).__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        ### initializeing the numerical front ###
        self.act_fun = []
        self.depth = len(width) - 1
        self.width = width

        for l in range(self.depth):
            # splines
            sp_batch = KANLayer(width[l], width[l + 1], grid_range)
            self.act_fun.append(sp_batch)

        self.act_fun = nn.ModuleList(self.act_fun)

    def reg(self,acts_scale):
        lamb_l1=1. 
        lamb_entropy=2.

        def nonlinear(x, th=1e-16, factor=1):
            return (x < th) * x * factor + (x > th) * (x + (factor - 1) * th)

        reg_ = 0.
        for i in range(len(acts_scale)):
            vec = acts_scale[i].reshape(-1, )

            p = vec / torch.sum(vec)
            l1 = torch.sum(torch.abs(nonlinear(vec))).item()
            entropy = - torch.sum(torch.abs(p * torch.log2(p + 1e-4))).item()
            reg_ += lamb_l1 * l1 + lamb_entropy * entropy  # both l1 and entropy

        return reg_


    def forward(self, x):
        x = x.view(-1, self.width[0])
        self.acts_scale = []
        for l in range(self.depth):
            x = self.act_fun[l](x)
            self.acts_scale.append(x)
        return x
    
