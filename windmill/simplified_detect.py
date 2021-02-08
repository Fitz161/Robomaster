import cv2
import numpy as np

PATH = r'D:\photos\robomaster_photo\2.jpg'
img = cv2.imread(PATH, 1)
gray_img = cv2.imread(PATH, 0)
blue, green, red = cv2.split(img)

print('image shape:', img.shape)
red_img = red - blue # 采用红色通道与蓝色通道的差作为待处理图像

# 去除背景
width, height = red_img.shape
for x in range(width):
    for y in range(height):
        if red_img[x, y] < 30:
            red_img[x, y] = 255

# 二值化
bin_img = cv2.threshold(red_img, 200, 255, cv2.THRESH_BINARY)[1]

# 反转二值图颜色
bin_img = np.where(bin_img > 180, 0, 255)

# 对其中白色部分先膨胀，再腐蚀，去除内部噪声
kernel = np.ones((2, 2), np.uint8)

bin_img = bin_img.astype(np.uint8) #int32转成uint8
dilation = cv2.dilate(bin_img, kernel, iterations=2)
erosion = cv2.erode(dilation, kernel, iterations=2)

# 漫水填充，获取图形内部部分
h, w = erosion.shape
mask = np.zeros([h + 2, w + 2], np.uint8)
cv2.floodFill(erosion, mask, (30, 30), 255, 10, 10,
              cv2.FLOODFILL_FIXED_RANGE)  # 5，6两个参数的值无影响


# 中值滤波,去除噪音
blured_img = cv2.medianBlur(erosion, ksize=9)  # ksize取7，9时效果较好

# 边缘检测,获得图形边缘
edge = cv2.Canny(blured_img, 0, 255)  # 后两参数无影响

# 提取轮廓
contours, hierarchy = cv2.findContours(
    edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 根据每个轮廓长度判断是击打区还是支架
length_list = []
for i in range(len(contours)):
    length_list.append(int(cv2.arcLength(contours[i], True)))
# 通过取平均数的方式去掉大的轮廓
len_list = np.array(length_list)
index_list = np.where(len_list < np.mean(len_list))  # 保留元素的索引数组

# 将轮廓拟合为矩形

appro_list, sum_list = [], []
# 获取拟合到的多边形的顶点坐标
for i in index_list[0]:
    appro_list.append(
        cv2.approxPolyDP(
            contours[i],
            epsilon=6,
            closed=True))  # 第二个参数epsilon越低，表示拟合精度越高

# 剔除非四边形
for arr in appro_list:
    if not arr.size == 8:
        appro_list.remove(arr)
    else:
        sum_list.append(np.sum(arr.ravel()))  # np.ravel将矩阵flat成一维，np.sum加和

print('number of rectangles:', len(appro_list))
print('sum of coordinates of vertex of rectangles:', sum_list)

# 对相邻近的矩形去重
sum_list2 = list(sum_list)  # 用list构造新列表进行deep copy
appro_list2 = []
# for range循环中不要对自身进行修改，防止下标过界等，用while循环
while sum_list:  # 列表不为空，取最后一个元素
    check_num = sum_list.pop()
    for num in sum_list:
        if abs(check_num - num) < 50:  # 阈值
            sum_list.remove(num)  # 去除列表所有与check_num相近的值
    # 将check_num对应元素（通过下标获得）放到新列表中
    #print(check_num, sum_list2.index(check_num))
    appro_list2.append(appro_list[sum_list2.index(check_num)])

#print(len(appro_list), len(appro_list2))

for appro in appro_list2:
    for j in range(4):
        # 通过画线方式来在原图像中绘制矩形
        cv2.line(img, tuple(appro[j, 0]), tuple(
            appro[(j + 1) % 4, 0]), (255, 0, 0), 2)

cv2.imshow('', img)
cv2.waitKey(0)
cv2.imwrite('1.jpg', img)