import os
import cv2
import torch


# 数据源：http://www.nlpr.ia.ac.cn/pal/trafficdata/detection.html
# 该数据源中的路标检测仅仅提供了位置，没有类别；
# 有个别几张图片名字相同，格式不同，导致对应的label.txt名字与图片部队与，大概有3张左右，直接把这3张删掉了

def max_min(l):
    m = max(l)
    n = min(l)
    return m, n


def process2train_data():
    """
    基于原始数据的标注，生成yolo v3的数据标注
    """
    img_folder = 'D:\\myself\\自动驾驶\\路标位置检测\\train_image\\'
    label_folder = 'D:\\myself\\自动驾驶\\路标位置检测\\label\\'
    out_folder = "D:\\myself\\自动驾驶\\路标位置检测\\代码\\myself\\labels\\"

    images = os.listdir(img_folder)
    images.sort()
    labels = os.listdir(label_folder)
    labels.sort()
    count = 0
    for img, label in zip(images, labels):
        count += 1
        if count % 10 == 0:
            print("count = ", count)
        img_info = cv2.imread(img_folder + img)
        height, width = img_info.shape[0], img_info.shape[1]
        print(width, height)
        with open(label_folder + label) as label_file:
            with open(out_folder + label, "w") as out_file:
                for line in label_file:
                    if len(line) < 15:
                        continue
                    coord = list(map(lambda x: int(x), line.strip("\n").split(",")))
                    max_x, min_x = max_min(coord[0::2])
                    max_y, min_y = max_min(coord[1::2])
                    center_x = (max_x + min_x) / 2
                    center_y = (max_y + min_y) / 2
                    obj_width = max_x - min_x
                    obj_height = max_y - min_y

                    out_file.write(
                        " ".join(["0",
                                  "{:.4f}".format(center_x / width),
                                  "{:.4f}".format(center_y / height),
                                  "{:.4f}".format(obj_width / width),
                                  "{:.4f}".format(obj_height / height)]))
                    out_file.write("\n")


"""
train 逻辑：
---------------myself.yaml--------------
train: ../myself-test/images
val: ../myself-test/images

nc: 1
names: ['traffic-sign']
----------------------------------------
训练环境：https://colab.research.google.com/github/ultralytics/yolov3/blob/master/tutorial.ipynb
训练命令：python train.py --img 640 --batch 16 --epochs 300 --data myself.yaml --weights '' --cfg yolov3-tiny.yaml --nosave --cache
"""
if __name__ == '__main__':
    # 将自己训练的模型命名为yolov3_tiny.pt 与此代码放到同一个目录下
    model = torch.hub.load('ultralytics/yolov3', 'yolov3_tiny')
    dir = "../../data/traffic_detect/"
    files = os.listdir(dir)
    for file in files:
        results = model(dir + file)
        results.show()  # or .show(), .save()
