# -*- coding: utf-8 -*-
import cv2
import torch
import torch.nn as nn
import numpy as np
from statistics import mode


# 人脸数据归一化,将像素值从0-255映射到0-1之间
def preprocess_input(images):
    """ preprocess input by substracting the train mean
    # Arguments: images or image of any shape
    # Returns: images or image with substracted train mean (129)
    """
    images = images/255.0
    return images



class VGG(nn.Module):
    def __init__(self, *args):
        super(VGG, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0],-1)


def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 这里会使宽高减半
    return nn.Sequential(*blk)
    
    
conv_arch = ((2, 1, 32), (3, 32, 64), (3, 64, 128))
# 经过5个vgg_block, 宽高会减半5次, 变成 224/32 = 7
fc_features = 128 * 6* 6 # c * w * h
fc_hidden_units = 4096 # 任意

def vgg(conv_arch, fc_features, fc_hidden_units):
    net = nn.Sequential()
    # 卷积层部分
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        # 每经过一个vgg_block都会使宽高减半
        net.add_module("vgg_block_" + str(i+1), vgg_block(num_convs, in_channels, out_channels))
    # 全连接层部分
    net.add_module("fc", nn.Sequential(
                                 VGG(),
                                 nn.Linear(fc_features, fc_hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(fc_hidden_units, fc_hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(fc_hidden_units, 7)
                                ))
    return net


#opencv自带的一个面部识别分类器
detection_model_path = 'model/haarcascade_frontalface_default.xml'

classification_model_path = 'model/model_vgg.pkl'

# 加载人脸检测模型
face_detection = cv2.CascadeClassifier(detection_model_path)

# 加载表情识别模型
emotion_classifier = torch.load(classification_model_path)


frame_window = 10

#表情标签
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

emotion_window = []

# 调起摄像头，0是笔记本自带摄像头
video_capture = cv2.VideoCapture(0)
# 视频文件识别
# video_capture = cv2.VideoCapture("video/example_dsh.mp4")
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.startWindowThread()
cv2.namedWindow('window_frame')

while True:
    # 读取一帧
    _, frame = video_capture.read()
    frame = frame[:,::-1,:]#水平翻转，符合自拍习惯
    frame = frame.copy()
    # 获得灰度图，并且在内存中创建一个图像对象
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 获取当前帧中的全部人脸
    faces = face_detection.detectMultiScale(gray,1.3,5)
    # 对于所有发现的人脸
    for (x, y, w, h) in faces:
        # 在脸周围画一个矩形框，(255,0,0)是颜色，2是线宽
        cv2.rectangle(frame,(x,y),(x+w,y+h),(84,255,159),2)

        # 获取人脸图像
        face = gray[y:y+h,x:x+w]

        try:
            # shape变为(48,48)
            face = cv2.resize(face,(48,48))
        except:
            continue

        # 扩充维度，shape变为(1,48,48,1)
        #将（1，48，48，1）转换成为(1,1,48,48)
        face = np.expand_dims(face,0)
        face = np.expand_dims(face,0)

        # 人脸数据归一化，将像素值从0-255映射到0-1之间
        face = preprocess_input(face)
        new_face=torch.from_numpy(face)
        new_new_face = new_face.float().requires_grad_(False)
        
        # 调用我们训练好的表情识别模型，预测分类
        emotion_arg = np.argmax(emotion_classifier.forward(new_new_face).detach().numpy())
        emotion = emotion_labels[emotion_arg]

        emotion_window.append(emotion)

        if len(emotion_window) >= frame_window:
            emotion_window.pop(0)

        try:
            # 获得出现次数最多的分类
            emotion_mode = mode(emotion_window)
        except:
            continue

        # 在矩形框上部，输出分类文字
        cv2.putText(frame,emotion_mode,(x,y-30), font, .7,(0,0,255),1,cv2.LINE_AA)

    try:
        # 将图片从内存中显示到屏幕上
        cv2.imshow('window_frame', frame)
    except:
        continue

    # 按q退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
