import os
import time
import cv2
import numpy as np

show_process = False
is_video = True

"""
二值图中出现扇叶过大且相交时,调小dilation次数,或调大erosion次数     mark1
扇叶和装甲板中心个数均为奇数时可能需要调整比值width/height          mark2
识别待击打装甲板失败时可能需要调整阈值width*num                    mark3
"""

def distance(point1, point2):
    """
    return the distance of two points
    :param point1: (x, y) or [x, y] representing coordinates of the point
    :param point2: (x, y) or [x, y]
    :return: float number
    """
    return ((point1[0] - point2[0]) ** 2 +
            (point1[1] - point2[1]) ** 2) ** (1 / 2)


def handle_one_frame(frame):
    """
    handle and process one image
    :param frame: np.ndarray object with three channels
    :return: None
    """
    start = time.clock()

    blue, green, red = cv2.split(frame)

    print('image shape:', frame.shape)
    red_img = blue - red # 采用红色通道与蓝色通道的差作为待处理图像

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

    '''mark1'''

    # 对其中白色部分先膨胀，再腐蚀，去除内部噪声，闭运算
    kernel = np.ones((4, 4), np.uint8)  # 核越大则图形越膨胀变粗，内部噪点也越少，需避免相交

    bin_img = bin_img.astype(np.uint8)  # int32转成uint8
    # 膨胀操作
    dilation = cv2.dilate(bin_img, kernel, iterations=1) # 二值图中出现扇叶过大且相交时，调小dilation次数
    # 腐蚀操作
    erosion = cv2.erode(dilation, kernel, iterations=1) # 或调大erosion次数

    if show_process:
        cv2.imshow('', erosion)
        cv2.waitKey(0)

    # 漫水填充，获取图形内部部分
    h, w = erosion.shape
    mask = np.zeros([h + 2, w + 2], np.uint8)
    cv2.floodFill(erosion, mask, (30, 30), 255, 10, 10,
                  cv2.FLOODFILL_FIXED_RANGE)  # 5，6两个参数的值无影响
    if show_process:
        cv2.imshow('', erosion)
        cv2.waitKey(0)

    # 边缘检测,获得图形边缘
    edge = cv2.Canny(erosion, 0, 255)  # 后两参数无影响

    # 提取轮廓
    contours, hierarchy = cv2.findContours(
        edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    blade_width = []  # 用于求扇叶平均宽度，作为下方合并相邻扇叶的阈值

    if show_process:
        white_canvas = np.ones([edge.shape[0], edge.shape[1], 3],
                               dtype=np.uint8)*255
        cv2.drawContours(white_canvas, contours, -1, 0, 2)
        print('contours', len(contours))
        cv2.imshow('', white_canvas)
        cv2.waitKey(0)
        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            cv2.rectangle(white_canvas, (x, y), (x + w, y + h), (0, 255, 255), 1)
        cv2.imshow('', white_canvas)
        cv2.waitKey(0)

    '''mark2'''

    # 分离扇叶和装甲板轮廓和对应中心点
    armor_list, armor_center = [], []
    blade_list, blade_center = [], []
    for contour in contours:
        # 返回轮廓最小外接矩形的 中心(x,y), (宽,高), 旋转角度
        rectangle = cv2.minAreaRect(contour)
        width, height = sorted(rectangle[1])  # 宽和高的大小不确定，进行排序
        if width < 5 or height < 10:  # 剔除过小的轮廓
            continue
        # print(length[1] / length[0])   # 1.6-2.2  3.8-6
        if height / width > 2.4: # 扇叶和装甲板中心个数均为奇数时可能需要调整这个
            blade_width.append(width)
            blade_list.append(cv2.boxPoints(rectangle))
            blade_center.append(rectangle[0])
        else:
            # 获取该矩形的四个顶点坐标
            armor_list.append(cv2.boxPoints(rectangle))
            armor_center.append(rectangle[0])

    print('blade_center', len(blade_center), blade_center)
    print('armor_center', len(armor_center), armor_center)

    # 通过画线方式来在原图像中绘制矩形
    for armor in armor_list:
        for j in range(4):
            cv2.line(frame, tuple(armor[j]), tuple(
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
            if abs(center[0] - point[0]) < 5 and abs(center[1] - point[1]) < 5:
                blade_center.remove(point)
        filtered_blade_center.append(center)

    width = int(sum(blade_width) / len(blade_width)) if blade_width else None
    print('width', width)
    print('filtered_blade_center', len(filtered_blade_center), filtered_blade_center)
    print('filtered_armor_center', len(filtered_armor_center), filtered_armor_center)

    '''mark3'''

    # 获取相邻两片扇叶的中心, 并放到blade_center中
    while filtered_blade_center:
        center = filtered_blade_center.pop()
        for point in filtered_blade_center:
            # 用二倍扇叶长度做阈值
            if abs(center[0] - point[0]) < width * \
                    3 and abs(center[1] - point[1]) < width * 3: # 识别待击打装甲板失败时可能需要调整这个
                blade_center.append(
                    ((center[0] + point[0]) / 2,
                     (center[1] + point[1]) / 2))
                filtered_blade_center.remove(point)
                break

    print('filtered_blade_center', len(blade_center), blade_center)
    print('filtered_armor_center', len(filtered_armor_center), filtered_armor_center)

    filtered_armor_center = [tuple(map(int, num_tuple))
                             for num_tuple in filtered_armor_center]  # 浮点数转整数

    blade_center_num = len(blade_center)


    if not blade_center_num:  # 无扇叶时，直接将唯一的装甲板识别为待打击对象
        if not filtered_armor_center:
            return
        cv2.circle(
            frame,
            filtered_armor_center[0],
            4,
            (255, 255, 0),
            thickness=-1)  # thickness为负表示填充圆形
    else:  # 有扇叶时，已知相邻两扇叶的中心，计算与每个装甲板中心的距离，最近的即为配对的装甲板，筛选剩的即为待击打对象
        if not len(filtered_armor_center) == 5:
            for blade_point in blade_center:
                distance_list = []
                for armor_point in filtered_armor_center:
                    distance_list.append(distance(blade_point, armor_point))
                # 找到离扇叶中心最近的装甲板并将其删除
                if distance_list:
                    index = distance_list.index(min(distance_list))
                    del filtered_armor_center[index]
            # 画出最后剩下的一个装甲板的位置
            if filtered_armor_center:
                cv2.circle(
                    frame,
                    filtered_armor_center[0],
                    4,
                    (255, 255, 0),
                    thickness=-1)  # thickness为负表示填充圆形

    end = time.clock()
    print('process time', end - start)

    cv2.imshow('', frame)
    cv2.waitKey(1000)


if is_video:
    video_path = r'D:/photos/robomaster/red2.mp4' #可以读取mp4视频，无法读取avi视频
    #video_path = "‪D:/photos/robomaster/red001.avi"  # 视频名要加编号，否则报错
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)		# 视频的帧率FPS
    total_frame = capture.get(cv2.CAP_PROP_FRAME_COUNT)  # 视频的总帧数
    print('fps', fps, 'total frame', total_frame)


    for i in range(int(total_frame)):
        # read 相当于grab和retrieve，返回retrieve和frame
        ret:bool = capture.grab() # 获得一帧
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # 每隔固定帧后处理一帧
        if i % int(fps) == 0:
            ret, frame = capture.retrieve() #解码并返回一帧
            if ret:
                handle_one_frame(frame)
            else:
                print("Error retrieving frame from movie!")
                break
    capture.release() # 释放视频
    cv2.destroyAllWindows()

else:
    path = 'D:/photos/robomaster/'

    file_names, image_paths = list(os.walk(path))[0][2], []
    for file_name in file_names:
        if file_name[-3:] == 'jpg' or file_name[-3:] == 'png':
            image_paths.append(path + file_name)

    print(file_names, image_paths)

    for image_path in image_paths:
        print(image_path)
        handle_one_frame(cv2.imread(image_path, cv2.IMREAD_COLOR))

