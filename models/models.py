# @Time    : 2023/11/7
# @Author  : SXS
# @Github  : https://github.com/SXS-PRIVATE/HNH
import torch
import math
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import AlexNet_Weights


class ImgNet(nn.Module):
    def __init__(self, code_len):
        super(ImgNet, self).__init__()
        self.alexnet = torchvision.models.alexnet(weights=AlexNet_Weights.DEFAULT)
        self.alexnet.classifier = nn.Sequential(*list(self.alexnet.classifier.children())[:6])
        self.fc_encode = nn.Linear(4096, code_len)
        self.alpha = 1.0


    def forward(self, x):
        x = self.alexnet.features(x)
        x = x.view(x.size(0), -1)
        feat = self.alexnet.classifier(x)
        hid = self.fc_encode(feat)
        code = F.tanh(hid)

        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)


class TxtNet(nn.Module):
    def __init__(self, code_len, txt_feat_len):
        super(TxtNet, self).__init__()
        self.fc1 = nn.Linear(txt_feat_len, 8192)
        self.fc2 = nn.Linear(8192, 4096)
        self.fc3 = nn.Linear(4096, code_len)
        self.alpha = 1.0

    def forward(self, x):
        x = F.relu(self.fc1(x))
        feat = F.relu(self.fc2(x))
        hid = self.fc3(feat)
        code = F.tanh(self.alpha * hid)
        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)
