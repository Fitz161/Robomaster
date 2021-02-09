import cv2
import numpy as np

PATH = r'D:/photos/robomaster/2.jpg'
img = cv2.imread(PATH, 1)
gray_img = cv2.imread(PATH, 0)
blue, green, red = cv2.split(img)

print('image shape:', img.shape)
red_img = red - blue # 采用红色通道与蓝色通道的差作为待处理图像

# 去除背景
width, height = red_img.shape
for x in range(width):
    for y in range(height):
        if red_img[x, y] < 50:  # 根据需要调整阈值，越大时去除外部噪点效果越好，但图形越浅
            red_img[x, y] = 255

# 二值化
bin_img = cv2.threshold(red_img, 200, 255, cv2.THRESH_BINARY)[1]

# 反转二值图颜色
bin_img = np.where(bin_img > 180, 0, 255)

# 对其中白色部分先膨胀，再腐蚀，去除内部噪声
kernel = np.ones((4, 4), np.uint8)  #核越大则图形越膨胀变粗，内部噪点也越少，需避免相交

bin_img = bin_img.astype(np.uint8) #int32转成uint8
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

# 分离扇叶和装甲板轮廓
armor_list, armor_center = [], []
blade_list, blade_center = [], []
for contour in contours:
    #返回轮廓最小外接矩形的 中心(x,y), (宽,高), 旋转角度
    rectangle = cv2.minAreaRect(contour)
    armor_center.append(rectangle[0])
    length = sorted(rectangle[1])  #宽和高的大小不确定
    #print(length[1] / length[0])   # 1.6  3.8
    if length[1] / length[0] > 2:
        blade_list.append(cv2.boxPoints(rectangle))
    else:
        # 获取该矩形的四个顶点坐标
        armor_list.append(cv2.boxPoints(rectangle))

# 通过画线方式来在原图像中绘制矩形
for box in armor_list:
    for j in range(4):
        cv2.line(img, tuple(box[j]), tuple(
            box[(j + 1) % 4]), (255, 0, 0), 2)

# 对矩形中心去重
filtered_center = []
while armor_center:
    center = armor_center.pop()
    for point in armor_center:
        if abs(center[0] - point[0]) > 10 or abs(center[1] - point[1]) > 10:
            armor_center.remove(point)
    filtered_center.append(center)

print(filtered_center)

# 中值滤波,去除噪音
# blured_img = cv2.medianBlur(bin_img, ksize=5)  # 只能取奇数, 越大图像噪点越少
#
# print(cv2.HoughCircles(blured_img, cv2.HOUGH_GRADIENT, dp=2, minDist=100))#, minRadius=5, maxRadius=10))

cv2.imshow('', img)
cv2.waitKey(0)
