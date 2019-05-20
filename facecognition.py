# -*- codeing: utf-8 -*-
import cv2
import dlib
import os
import sys
import random

#输出路径，同目录下的my_faces文件夹
output_dir = './my_faces'
#保存的图片为64*64
size = 64

#如果该路径不存在则创建一个
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 改变图片的亮度与对比度
def relight(img, light=1, bias=0):
    #shape[1]是图片像素的列数（即width）
    w = img.shape[1]
    #shape[0]是图片像素的行数（即height）
    h = img.shape[0]
    #遍历每个像素，img[行，列，BGR三元组的某一项]
    for i in range(0,w):
        for j in range(0,h):
            for c in range(3):
                tmp = int(img[j,i,c]*light + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j,i,c] = tmp
    return img


#使用dlib自带的frontal_face_detector作为我们的特征提取器
detector = dlib.get_frontal_face_detector()
# 打开摄像头 参数为输入流，可以为摄像头或视频文件
camera = cv2.VideoCapture(0)

#图片的开始下标记
index = 1
while True:
    if (index <= 10100):
        print('Being processed picture %s' % index)
        # 从摄像头读取照片
        success, img = camera.read()
        # 转为灰度图片
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 使用detector进行人脸检测
        dets = detector(gray_img, 1)

        #对于一个可迭代的/可遍历的对象，enumerate将其组成一个索引序列
        #利用它可以同时获得索引(i)和值(d)
        #left：人脸左边距离图片左边界的距离 ；right：人脸右边距离图片左边界的距离
        #top：人脸上边距离图片上边界的距离 ；bottom：人脸下边距离图片上边界的距离
        for i, d in enumerate(dets):
            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0

            #提取img中的一部分img[y:y+h,x:x+w]
            face = img[x1:y1,x2:y2]

            #调整图片的对比度与亮度， 对比度与亮度值都取随机数，这样能增加样本的多样性
            #uniform()用于生成一个指定范围内的实数
            #randint()用于生成一个指定范围内的整数
            face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))

            #图像缩放至64*64
            face = cv2.resize(face, (size,size))

            cv2.imshow('image', face)

            cv2.imwrite(output_dir+'/'+str(index)+'.jpg', face)

            index += 1

        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break
    else:
        print('Finished!')
        break
#while循环结束
