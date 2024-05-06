import torch,os
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms


root = os.getcwd()
batch_size = 100
# 加载数据时即将其转换为tensor并归一化
train_dataset = dsets.MNIST(root, train=True, download=True,
                            transform=transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))]))
test_dataset = dsets.MNIST(root, train=False, download=True,
                            transform=transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))]))

# 生成训练loader和测试loader
train_loader = DataLoader(dataset=train_dataset,
                           batch_size=batch_size,
                             shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                           batch_size=batch_size,
                            shuffle=True)