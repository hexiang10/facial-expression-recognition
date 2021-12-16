# model.py
import torch.nn as nn
import torch

'''
class VGG(nn.Module):
创建一个类，类名为VGG，VGG继承于 nn.module父类，实现在搭建网络过程中所需要的用到的一些网络层结构
torch.nn.module:所有神经网络模块的基类；模型也应该继承此类；模块也可以包含其他模块，从而可以将它们嵌套在树结构中。
'''
class VGG(nn.Module):
    '''
    def init(self, features, num_classes=1000, init_weights=False):
    在VGG中定义__init__函数
    features:在之后的代码中会生成网络特征结构
    num_classes=1000:将要进行的分类的个数
    init_weights=False:init_weights是一个bool型的参数；用于决定是否对网络进行权重的初始化，
    如果bool值为True，则进入到定义好的初始化权重中  

    '''
    def __init__(self, features, num_classes=1000, init_weights=False):
        '''
        super(VGG, self).init()
        在多重继承中，调用父类的过程中可能出现一系列问题，super在涉及到多继承时使用

        '''
        super(VGG, self).__init__()
        self.features = features #用于提取图像的特征

        '''
        self.classifier = nn.Sequential
        将一系列层结构进行打包，组成一个新的结构，生成分类网络结构，nn.Sequential在网络层次多的时候，可以用于代码的精简。
        torch.nn.Squential：一个连续的容器。模块将按照在构造函数中传递的顺序添加到模块中。或者，也可以传递模块的有序字典。

        nn.Dropout(p=0.5)
        随机失活一部分神经元，用于减少过拟合，默认比例为0.5，仅用于正向传播。

        nn.Linear(51277, 4096)
        在进行全连接之间需要将输入的数据展开成为一位数组。51277是展平之后得到的一维向量的个数。4096是全连接层的结点个数。

        nn.ReLU(True)
        定义ReLU函数
        relu函数：f(x) = Max (0 ,x)

        nn.Dropout(p=0.5)
        随机失活一部分神经元

        nn.Linear(4096, 4096)
        定义第二个全连接层，输入为4096，本层的结点数为4096

        nn.ReLU(True):
        定义全连接层

        nn.Linear(4096, num_classes)
        最后一个全连接层。num_classes:分类的类别个数。

        if init_weights:
        是否对网络进行参数初始化

        self._initialize_weights()
        若结果为真，则进入到_initialize_weights函数中
        '''

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

        '''
        def forward(self, x):
        定义前向传播过程
        x为输入的图像数据(x代表输入的数据)

        x = self.features(x)
        将x输入到features，并赋值给x

        x = torch.flatten(x, start_dim=1)
        进行展平处理，start_dim=1，指定从哪个维度开始展平处理，因为第一个维度是batch，不需要对它进行展开，所以从第二个维度进行展开。

        x = self.classifier(x)
        展平后将特征矩阵输入到事先定义好的分类网络结构中。

        return x
        返回值为x

        '''

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

        '''
        def _initialize_weights(self):
        定义初始化权重函数

        for m in self.modules():
        用m遍历网络的每一个子模块，即网络的每一层

        if isinstance(m, nn.Conv2d):
        若m的值为 nn.Conv2d,即卷积层
        Conv2d:对由多个输入平面组成的输入信号应用2D卷积。
        # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        凯明初始化方法

        nn.init.xavier_uniform_(m.weight)
        用xavier初始化方法初始化卷积核的权重

        if m.bias is not None:
        nn.init.constant_(m.bias, 0)
        若偏置不为None，则将偏置初始化为0

        elif isinstance(m, nn.Linear):
        若m的值为nn.Linear,即池化层
        # nn.init.normal_(m.weight, 0, 0.01)
        用一个正太分布来给权重进行赋值，0为均值，0.01为方差

        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)
        对权重进行赋值，并且将偏置初始化为0

        '''



    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

'''
def make_features(cfg: list):
提取特征网络结构，
cfg.list:传入配置变量，只需要传入对应配置的列表

layers = []
空列表，用来存放所创建的每一层结构

in_channels = 3
输入数据的深度，因为输入的是彩色RGB图像，通道数为3

for v in cfg:
遍历cfg，即遍历配置变量列表

if v == “M”:
layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
若为最大池化层，创建池化操作，并为卷积核大小设置为2，步距设置为2，并将其加入到layers

else:
conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
layers += [conv2d, nn.ReLU(True)]
in_channels = v
否则创建卷积操作，定义输入深度，配置变量，卷积核大小为3，padding操作为1，并将Conv2d和ReLU激活函数加入到layers列表中
in_channels = v 经卷积后深度为v

*return nn.Sequential(layers)
将列表通过非关键字参数的形式传入进去，*代表通过非关键字形式传入

'''
def make_features(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)

'''
cfgs为字典文件，其中vgg11，vgg13，vgg16，vgg19为字典的key值，对应的value为模型的配置文件

以vgg11为例
64 卷积层卷积核个数
'M', 池化层的结构
128, 卷积层卷积核个数
'M', 池化层的结构
256, 卷积层卷积核个数
256, 卷积层卷积核个数
'M', 池化层的结构
512, 卷积层卷积核个数
512, 卷积层卷积核个数
'M', 池化层的结构
512, 卷积层卷积核个数
512, 卷积层卷积核个数
'M' 池化层的结构
'''
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

'''
**def vgg(model_name=“vgg16”, kwargs):
定义函数vgg，通过此函数实例化锁给定的配置模型，
**kwargs:定义vgg函数时传入的字典变量，包含所需的分类个数以及是否初始化权重的布尔变量

try:
cfg = cfgs[model_name]
except:
print(“Warning: model number {} not in cfgs dict!”.format(model_name))
exit(-1)
cfg = cfgs[model_name]:将vgg16的key值传入到字典中，得到所对应的配置列表，命名为cfg

**model = VGG(make_features(cfg), kwargs)
return model
通过VGG函数来说实例化VGG网络
make_features(cfg):将cfg中的数据传入到make_features中
**kwargs:一个可变长度的字典变量
'''

def vgg(model_name="vgg16", **kwargs):
    try:
        cfg = cfgs[model_name]
    except:
        print("Warning: model number {} not in cfgs dict!".format(model_name))
        exit(-1)
    model = VGG(make_features(cfg), **kwargs)
    return model


# train.py
import torch.nn as nn
from torchvision import transforms, datasets
import json
import os
import torch.optim as optim
from model import vgg
import torch

'''
device用于指定我们在训练过程中所使用的设备，打印使用的设备信息。
if torch.cuda.is_available() else "cpu"
当前有可使用的GPU设备，默认使用当前设备列表的第一块GPU设备，否则使用CPU

'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

'''
data_transform
数据预处理函数

transforms.Compose
将我们所使用的预处理方法打包成为一个整体

transforms.RandomResizedCrop(224),
随机裁剪为224*224像素大小
RandomResizedCrop为随机裁剪
注:预处理第一步会将RGB三个通道分别减去[123.68,116.78,103.94]这三个值，这三个值对应的是ImageNet的所有图片RGB三个通道值的均值，这里我们进行的是从头开始训练。若基于迁移学习的方式训练就需要减去这三个值，因为它预训练的模型是基于ImageNet数据集进行训练的，所以使用预训练模型的时候，需要将它的图像减去对应的RGB值分量。

transforms.RandomHorizontalFlip(),
随机翻转（在水平的方向上随机饭翻转）

transforms.ToTensor(),
将数据转化为Tensor，将原本的取值范围0-255转化为0.1~1.0，并且将原来的序列（H，W，C）转化为（C，H，W）

transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]),
标准化处理，Normalize使用均值和标准差对Tensor进行标准化

'''

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

'''
这一部分为获取我们训练所需要使用的数据集，
“…/…”:…表示返回上一层目录，…/…表示返回上上层目录
(os.getcwd() 获取当前文件所在目录
os.path.join 将两个路径连在一起

'''

data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
image_path = data_root + "/data_set/flower_data/"  # flower data set path

'''
root=image_path+“train”,
加载数据集的路径

transform
数据预处理

data_transform[“train”]
将训练集传入，返回训练集对应的处理方式

train_num = len(train_dataset)
打印训练集的图片数量

'''

train_dataset = datasets.ImageFolder(root=image_path+"train",
                                     transform=data_transform["train"])
train_num = len(train_dataset)

'''
flower_list = train_dataset.class_to_idx
获取分类名称所对应的索引

cla_dict = dict((val, key) for key, val in flower_list.items())
遍历字典，将key val 值返回

json_str = json.dumps(cla_dict, indent=4)
通过json将cla_dict字典进行编码

with open('class_indices.json', 'w') as json_file:
json_file.write(json_str)
'class_indices.json', 将字典的key值保存在文件中，方便在之后的预测中读取它的信息

'''

# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 9

'''
train_loader = torch.utils.data.DataLoader(train_dataset,
batch_size=batch_size,shuffle=True,
num_workers=0)
通过DataLoader将train-dataset载入进来
shuffle=True, 随机参数，随机从样本中获取数据
num_workers=0 加载数据所使用的线程个数，线程增多，加快数据的生成速度，也能提高训练的速度，在Windows环境下线程数为0，在Linux系统下可以设置为多线程，一般设置为8或者16

validate_dataset = datasets.ImageFolder(root=image_path + “val”,
transform=data_transform[“val”])
载入测试集，transform为预处理

val_num = len(validate_dataset)
测试集文件个数

'''

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)

validate_dataset = datasets.ImageFolder(root=image_path + "val",
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0)

# test_data_iter = iter(validate_loader)
# test_image, test_label = test_data_iter.next()


'''
net = vgg(model_name=model_name, num_classes=6, init_weights=True)
调用vgg16，输入model的名称，指定所需要的vgg配置参数，以及权重的初始化，其中num_classes可以按照自己的数据集进行修改

net.to(device)
将网络指认到设备上

loss_function = nn.CrossEntropyLoss()
损失函数，使用的是针对多类别的损失交叉熵函数

optimizer = optim.Adam(net.parameters(), lr=0.0001)
优化器，优化对象为网络所有的可训练参数，且学习率设置为0.0001

'''
model_name = "vgg16"
net = vgg(model_name=model_name, num_classes=6, init_weights=True)
net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)
'''
定义最佳准确率，在训练过程中保存准确率最高的一次训练的模型
保存权重的路径

'''
best_acc = 0.0
save_path = './{}Net.pth'.format(model_name)

'''
for epoch in range(30):
设置迭代次数

net.train()
启用dropout，使用Dropout随机失活神将元的操作，但是我们只希望在训练的过程中如此，并不希望在预测的过程中起作用，通过net.tarin和net.eval来管理Dropout方法，这两个方法不仅可以管理Dropout，也可以管理BN方法

running_loss = 0.0
用于统计训练过程中的平均损失

for step, data in enumerate(train_loader, start=0):
遍历数据集，返回每一批数据data以及data对应的step

images, labels = data
将数据分为图像和标签

optimizer.zero_grad()
情况之间的梯度信息

outputs = net(images.to(device))
清空之前的梯度信息（清空历史损失梯度）

outputs = net(images.to(device))
将输入的图片引入到网络，将训练图像指认到一个设备中，进行正向传播得到输出，

loss = loss_function(outputs, labels.to(device))
将网络预测的值与真实的标签值进行对比，计算损失梯度

loss.backward()
optimizer.step()
误差的反向传播以及通过优化器对每个结点的参数进行更新

running_loss += loss.item()
将每次计算的loss累加到running_loss中

rate = (step + 1) / len(train_loader)
计算训练进度，当前的步数除以训练一轮所需要的总的步数

a = “*” * int(rate * 50)
b = “.” * int((1 - rate) * 50)
print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
用 * 和 . 来打印训练的进度

'''
for epoch in range(30):
    # train
    net.train()
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()

    '''
    net.eval()
    关闭Dropout

    with torch.no_grad():
    with 一个上下文管理器， torch.no_grad()之后的计算过程中，不计算每个结点的误差损失梯度

    for val_data in validate_loader:
    val_images, val_labels = val_data
    optimizer.zero_grad()
    outputs = net(val_images.to(device))
    遍历验证集；将数据划分为图片和对应的标签值，对结点进行参数的更新；指认到设备上并传入网络得到输出

    predict_y = torch.max(outputs, dim=1)[1]
    求得输出的最大值作为预测最有可能的类别

    acc += (predict_y == val_labels.to(device)).sum().item()
    predict_y == val_labels.to(device) 预测类别与真实标签值的对比，相同为1，不同为0
    item()获取数据，将Tensor转换为数值，通过sum()加到acc中，求和可得预测正确的样本数

    val_accurate = acc / val_num
    计算测试集额准确率

    if val_accurate > best_acc:
    best_acc = val_accurate
    torch.save(net.state_dict(), save_path)
    如果准确率大于历史最优的准确率，将当前值赋值给最优，并且保存当前的权重

    print('[epoch %d] train_loss: %.3f test_accuracy: %.3f' %
    (epoch + 1, running_loss / step, val_accurate))
    打印训练到第几轮，累加的平均误差，以及最优的准确率

    print('Finished Training')
    整个训练完成后打印提示信息
    '''
    # validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            optimizer.zero_grad()
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, val_accurate))

print('Finished Training')


# predict.py
import torch
from model import vgg
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json

'''
transforms.Compose
数据预处理

transforms.Resize((224, 224)),
将图片缩放到224224，因为vgg网络设置的传入图片大小的参数为224224

transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
转化为Tensor并进行标准化

img = Image.open(“C:\Users\39205\Desktop\1.jpg”)
plt.imshow(img)
预测图片路径以及图片的展示

'''

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 图片预处理，并且在最前面添加新的batch维度

# load image
img = Image.open("C:\\Users\\39205\\Desktop\\1.jpg")
plt.imshow(img)
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)

# read class_indict
try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

'''
model = vgg(model_name=“vgg16”, num_classes=6)
初始化网络

model_weight_path = "./vgg16Net.pth"
导入权重参数

model.load_state_dict(torch.load(model_weight_path))
载入网络模型

model.eval()
进入eval模式，即关闭Dropout

with torch.no_grad():
让pytorch不去跟踪变量的损失梯度

** output = torch.squeeze(model(img))**
model(img) 通过正向传播得到输出
torch.squeeze 将输出进行压缩，即压缩掉batch这个维度
output就是最终的输出

predict = torch.softmax(output, dim=0)
经过softmax函数将输出变为概率分布

** predict_cla = torch.argmax(predict).numpy()**
获取概率最大处所对应的索引值

print(class_indict[str(predict_cla)],predict[predict_cla].numpy())
输出预测的类别名称以及预测的准确率

plt.show()
展示进行分类的图片，这里只用了一张图片进行了预测，所以进行一下展示

'''

# create model
model = vgg(model_name="vgg16", num_classes=6)
# load model weights
model_weight_path = "./vgg16Net.pth"
model.load_state_dict(torch.load(model_weight_path))
model.eval()
with torch.no_grad():
    # predict class
    output = torch.squeeze(model(img))
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()
print(class_indict[str(predict_cla)],predict[predict_cla].numpy())
plt.show()


