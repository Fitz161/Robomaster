import cv2
import numpy as np

PATH = r'D:/photos/robomaster/5.png'
img = cv2.imread(PATH, 1)
gray_img = cv2.imread(PATH, 0)
blue, green, red = cv2.split(img)

print('image shape:', img.shape)
red_img = red - blue  # 采用红色通道与蓝色通道的差作为待处理图像

# 去除背景
width, height = red_img.shape
for x in range(width):
    for y in range(height):
        if red_img[x, y] < 50:  # 根据需要调整阈值，越大时去除外部噪点效果越好，但图形越浅
            red_img[x, y] = 255

# 二值化
bin_img = cv2.threshold(red_img, 200, 255, cv2.THRESH_BINARY)[1]

# 黑色部分膨胀，使图形更加清晰
kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(bin_img, kernel, iterations=1)

# 漫水填充，获取图形内部部分
h, w = erosion.shape
mask = np.zeros([h + 2, w + 2], np.uint8)
cv2.floodFill(erosion, mask, (30, 30), 0, 10, 10,
              cv2.FLOODFILL_FIXED_RANGE)  # 5，6两个参数的值无影响
# 参数newVal=255时结合下面筛选轮廓的最大矩形时 width和height的范围 可选中所有蓝色区域(包括中心，装甲板，扇叶）


# 边缘检测,获得图形边缘
edge = cv2.Canny(erosion, 0, 255)  # 后两参数无影响

# 提取轮廓
contours, hierarchy = cv2.findContours(
    edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

blade_width = []  # 用于求扇叶平均宽度，作为下方合并相邻扇叶的阈值

# 分离扇叶和装甲板轮廓和对应中心点
armor_list, armor_center = [], []
blade_list, blade_center = [], []
for contour in contours:
    # 返回轮廓最小外接矩形的 中心(x,y), (宽,高), 旋转角度
    rectangle = cv2.minAreaRect(contour)
    width, height = sorted(rectangle[1])  # 宽和高的大小不确定，进行排序
    if width < 5 or height < 15: # 剔除过小的轮廓
        continue
    # print(length[1] / length[0])   # 1.6  3.8
    if height / width > 2:
        blade_width.append(width)
        blade_list.append(cv2.boxPoints(rectangle))
        blade_center.append(rectangle[0])
    else:
        # 获取该矩形的四个顶点坐标
        armor_list.append(cv2.boxPoints(rectangle))
        armor_center.append(rectangle[0])


# 通过画线方式来在原图像中绘制矩形
for armor in armor_list:
    for j in range(4):
        cv2.line(img, tuple(armor[j]), tuple(
            armor[(j + 1) % 4]), (255, 0, 0), 2)

# 对矩形装甲板中心去重
filtered_armor_center = []
while armor_center:
    center = armor_center.pop()
    for point in armor_center:
        if abs(center[0] - point[0]) < 5 and abs(center[1] - point[1]) < 5:
            armor_center.remove(point)
    filtered_armor_center.append(center)

# 对扇叶中心去重
filtered_blade_center = []
while blade_center:
    center = blade_center.pop()
    for point in blade_center:
        if abs(center[0] - point[0]) < 10 and abs(center[1] - point[1]) < 10:
            blade_center.remove(point)
    filtered_blade_center.append(center)


width = int(sum(blade_width) / len(blade_width)) if blade_width else None

# 获取相邻两片扇叶的中心, 并放到blade_center中
while filtered_blade_center:
    center = filtered_blade_center.pop()
    for point in filtered_blade_center:
        # 用二倍扇叶长度做阈值
        if abs(center[0] - point[0]) < width * \
                2 and abs(center[1] - point[1]) < width * 2:
            blade_center.append(
                ((center[0] + point[0]) / 2,
                 (center[1] + point[1]) / 2))
            filtered_blade_center.remove(point)
            break

print('blade_center', blade_center)
print('filtered_armor_center', filtered_armor_center)

filtered_armor_center = [tuple(map(int, num_tuple))
                         for num_tuple in filtered_armor_center]  # 浮点数转整数


blade_center_num = len(blade_center)


def distance(point1, point2):
    """
    return the distance of two points
    :param point1: (x, y) or [x, y] representing coordinates of the point
    :param point2: (x, y) or [x, y]
    :return: float number
    """
    return ((point1[0] - point2[0]) ** 2 +
            (point1[1] - point2[1]) ** 2) ** (1 / 2)


if not blade_center_num:  # 无扇叶时，直接将唯一的装甲板识别为待打击对象
    cv2.circle(
        img,
        filtered_armor_center[0],
        4,
        (255, 255, 0),
        thickness=-1)  # thickness为负表示填充圆形
else:  # 有扇叶时，已知相邻两扇叶的中心，计算与每个装甲板中心的距离，最近的即为配对的装甲板，筛选剩的即为待击打对象
    for blade_point in blade_center:
        distance_list = []
        for armor_point in filtered_armor_center:
            distance_list.append(distance(blade_point, armor_point))
        # 找到离扇叶中心最近的装甲板并将其删除
        index = distance_list.index(min(distance_list))
        del filtered_armor_center[index]
    # 画出最后剩下的一个装甲板的位置
    if filtered_armor_center:
        cv2.circle(
            img,
            filtered_armor_center[0],
            4,
            (255, 255, 0),
            thickness=-1)  # thickness为负表示填充圆形


cv2.imshow('', img)
cv2.waitKey(0)
