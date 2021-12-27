import torch
import cv2 as cv
import torchvision.transforms as transforms
import torchvision
import numpy as np

# 参考：https://blog.csdn.net/qq_39071739/article/details/107940715
# 训练：https://zhuanlan.zhihu.com/p/365803576
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

preprocess = transforms.Compose([transforms.ToTensor()])

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

if __name__ == '__main__':
    frame = cv.imread("./file.png")
    blob = preprocess(frame)
    c, h, w = blob.shape
    input_x = blob.view(1, c, h, w)
    output = model(input_x)[0]
    boxes = output['boxes'].cpu().detach().numpy()
    scores = output['scores'].cpu().detach().numpy()
    labels = output['labels'].cpu().detach().numpy()
    masks = output['masks'].cpu().detach().numpy()
    index = 0
    color_mask = np.zeros((h, w, c), dtype=np.uint8)
    mv = cv.split(color_mask)
    for x1, y1, x2, y2 in boxes:
        if scores[index] > 0.5:
            cv.rectangle(frame, (np.int32(x1), np.int32(y1)),
                         (np.int32(x2), np.int32(y2)), (0, 255, 255), 1, 8, 0)
            mask = np.squeeze(masks[index] > 0.5)
            np.random.randint(0, 256)
            mv[2][mask == 1], mv[1][mask == 1], mv[0][mask == 1] = \
                [np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)]
            label_id = labels[index]
            label_txt = COCO_INSTANCE_CATEGORY_NAMES[label_id]
            cv.putText(frame, label_txt, (np.int32(x1), np.int32(y1)), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        index += 1
    color_mask = cv.merge(mv)
    result = cv.addWeighted(frame, 0.5, color_mask, 0.5, 0)
    cv.imwrite("demo_img13_test.png", result)
