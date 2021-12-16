import math
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

train_curve = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# param
batch_size = 100
n_class = 10
padding_size = 15
epoches = 10

train_dataset = torchvision.datasets.MNIST('./data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST('./data/', train=False, transform=transforms.ToTensor(), download=False)
train = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=5)
test = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=5)

def gelu(x):
  "Implementation of the gelu activation function by Hugging Face"
  return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class ResBlock(nn.Module):
  # 残差块
  def __init__(self, in_size, out_size1, out_size2):
    super(ResBlock, self).__init__()
    self.conv1 = nn.Conv2d(
        in_channels = in_size,
        out_channels = out_size1,
        kernel_size = 3,
        stride = 2,
        padding = padding_size
    )
    self.conv2 = nn.Conv2d(
        in_channels = out_size1,
        out_channels = out_size2,
        kernel_size = 3,
        stride = 2,
        padding = padding_size
    )
    self.batchnorm1 = nn.BatchNorm2d(out_size1)
    self.batchnorm2 = nn.BatchNorm2d(out_size2)
  
  def conv(self, x):
    # gelu效果比relu好呀哈哈
    x = gelu(self.batchnorm1(self.conv1(x)))
    x = gelu(self.batchnorm2(self.conv2(x)))
    return x
  
  def forward(self, x):
    # 残差连接
    return x + self.conv(x)

# resnet
class Resnet(nn.Module):
  def __init__(self, n_class = n_class):
    super(Resnet, self).__init__()
    self.res1 = ResBlock(1, 8, 16)
    self.res2 = ResBlock(16, 32, 16)
    self.conv = nn.Conv2d(
        in_channels = 16,
        out_channels = n_class,
        kernel_size = 3,
        stride = 2,
        padding = padding_size
    )
    self.batchnorm = nn.BatchNorm2d(n_class)
    self.max_pooling = nn.AdaptiveAvgPool2d(1)

  def forward(self, x):
    # x: [bs, 1, h, w]
    # x = x.view(-1, 1, 28, 28)
    x = self.res1(x)
    x = self.res2(x)
    x = self.max_pooling(self.batchnorm(self.conv(x)))

    return x.view(x.size(0), -1)

resnet = Resnet().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=resnet.parameters(), lr=1e-2, momentum=0.9)

# train
total_step = len(train)
sum_loss = 0
for epoch in range(epoches):
  for i, (images, targets) in enumerate(train):
    optimizer.zero_grad()
    images = images.to(device)
    targets = targets.to(device)
    preds = resnet(images)
    
    loss = loss_fn(preds, targets)
    sum_loss += loss.item()
    loss.backward()
    optimizer.step()
    if (i+1)%100==0:
      print('[{}|{}] step:{}/{} loss:{:.4f}'.format(epoch+1, epoches, i+1, total_step, loss.item()))
  train_curve.append(sum_loss)
  sum_loss = 0

# test
resnet.eval()
with torch.no_grad():
  correct = 0
  total = 0
  for images, labels in test:
    images = images.to(device)
    labels = labels.to(device)
    outputs = resnet(images)
    _, maxIndexes = torch.max(outputs, dim=1)
    correct += (maxIndexes==labels).sum().item()
    total += labels.size(0)
  
  print('in 1w test_data correct rate = {:.4f}'.format((correct/total)*100))

pd.DataFrame(train_curve).plot() # loss曲线


