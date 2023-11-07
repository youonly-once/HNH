
# @Time    : 2023/11/7
# @Author  : SXS
# @Github  : https://github.com/SXS-PRIVATE/HNH
import train
import torch

torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    train.run()

