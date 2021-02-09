import os
import cv2
import numpy as np


def handle_one_frame(frame):

    blue, green, red = cv2.split(frame)

    print('image shape:', frame.shape)
    red_img = red - blue  # 采用红色通道与蓝色通道的差作为待处理图像

    # 去除背景
    width, height = red_img.shape
    for x in range(width):
        for y in range(height):
            if red_img[x, y] < 50:   # 根据需要调整阈值，越大时去除外部噪点效果越好，但图形越浅
                red_img[x, y] = 255

    # 二值化
    bin_img = cv2.threshold(red_img, 200, 255, cv2.THRESH_BINARY)[1]

    # 反转二值图颜色
    bin_img = np.where(bin_img > 180, 0, 255)

    # 对其中白色部分先膨胀，再腐蚀，去除内部噪声
    kernel = np.ones((4, 4), np.uint8)

    bin_img = bin_img.astype(np.uint8)  # int32转成uint8
    dilation = cv2.dilate(bin_img, kernel, iterations=2)
    erosion = cv2.erode(dilation, kernel, iterations=2)

    # 漫水填充，获取图形内部部分
    h, w = erosion.shape
    mask = np.zeros([h + 2, w + 2], np.uint8)
    cv2.floodFill(erosion, mask, (30, 30), 255, 10, 10,
                  cv2.FLOODFILL_FIXED_RANGE)  # 5，6两个参数的值无影响


    # 边缘检测,获得图形边缘
    edge = cv2.Canny(erosion, 0, 255)  # 后两参数无影响

    # 提取轮廓
    contours, hierarchy = cv2.findContours(
        edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 通过轮廓外包矩形识别装甲板
    box_list = []
    for contour in contours:
        # 返回轮廓最小外接矩形的 中心(x,y), (宽,高), 旋转角度
        rectangle = cv2.minAreaRect(contour)
        length = sorted(rectangle[1])  # 宽和高的大小不确定，先排序方便之后比较

        if length[1] / length[0] > 2:   #装甲板1.6   支架3.8
            continue
        else:
            # 获取该矩形的四个顶点坐标
            box_list.append(cv2.boxPoints(rectangle))

    # 通过画线方式来在原图像中绘制矩形
    for box in box_list:
        for j in range(4):
            cv2.line(frame, tuple(box[j]), tuple(
                box[(j + 1) % 4]), (255, 0, 0), 2)


    cv2.imshow('', frame)
    cv2.waitKey(2000)




path = 'D:/photos/robomaster/'

file_names, image_paths = list(os.walk(path))[0][2], []
for file_name in file_names:
    if file_name[-3:] == 'jpg' or file_name[-3:] == 'png':
        image_paths.append(path + file_name)

print(file_names, image_paths)

for image_path in image_paths:
    print(image_path)
    handle_one_frame(cv2.imread(image_path, cv2.IMREAD_COLOR))


