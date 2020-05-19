# --------------------------------------------------------------------------------------------------------
# 2019/12/25
# src - standard_models.py
# md
# --------------------------------------------------------------------------------------------------------

"""
A collection of Pytorch models
"""
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class MNSIT_Simple(nn.Module):

    def __init__(self):
        super(MNSIT_Simple, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # 32x26x26
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # 64x24x24
        self.dropout1 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  # 128
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(128, 10)  # 10

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = th.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)  # = log(softmax(x))
        # output = th.exp(output) # probabilities
        # == output = F.softmax(x,dim=1)
        return output
