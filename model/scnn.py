'''
@Author: PANG Aoyu
@Date: 2023-03-31 
@Description: SCNN, use multi-channels to extract infos
@LastEditTime: 2023-03-31
'''
import gym
import numpy as np

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class SCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 64):
        """特征提取网络
        """
        super().__init__(observation_space, features_dim)
        net_shape = observation_space.shape # 每个 movement 的特征数量, 8 个 movement, 就是 (N, 8, K)
        # 这里 N 表示由 N 个 frame 堆叠而成
        print('observation_space.shape',observation_space.shape)
        self.view_conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(1, net_shape[-1]), padding=0), # N*8*K -> 128*8*1
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(8,1), padding=0), # 128*8*1 -> 256*1*1
            nn.ReLU(),
        )
        view_out_size = self._get_conv_out(net_shape)

        self.fc = nn.Sequential(
            nn.Linear(view_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim)
        )

    def _get_conv_out(self, shape):
        o = self.view_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))


    def forward(self, observations):
        

        batch_size = observations.size()[0] # (BatchSize, N, 8, K)
        observations=observations.view(batch_size,1,8,-1)# 只用了一片
        conv_out = self.view_conv(observations).view(batch_size, -1)
        return self.fc(conv_out)