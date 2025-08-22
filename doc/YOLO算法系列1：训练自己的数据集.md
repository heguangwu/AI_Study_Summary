# `YOLO` 算法

截止2025年，`YOLO`算法最新版本已经是`YOLO11`，本文重点讲述`YOLOv8`这个我目前用的较多的版本。

## `YOLO`数据目录

要训练一个基于`YOLO`的模型，首先就要准备数据，包括：

- 图片数据（`images`）：图片文件夹，要求图片大小一样，如不一样需先转换为一样大的图片
- 标签数据（`labels`）：标签文件夹，每一个图片文件对应一个同名的txt文件，文件每一行是一个标签数据，如图片中有多个标签，则有多行数据
  - 格式(字段以空格隔开)：类别ID 目标中心点X归一化坐标  目标中心点Y归一化坐标 宽度归一化值  高度归一化值
  - 示例：`0 0.3456 0.21 0.12 0.34`
- 元数据文件（`data.yaml`）：

```yaml
# 类别数
nc: 2
# 类别ID及对应的名称
names:
  - 0: person
  - 1: dog
#  数据集根目录
path: /data/yolo/dataset
train: train/images
val: valid/images
# test: images/test
```

在`train` 和`val`目录下都分别有`images`和`labels`目录分别对应图片和标签文件。

## 标注数据

目前有几种给数据打标签的方法：

- [`labelimg`](https://pypi.org/project/labelImg/)
  - 安装：`pip3 install labelImg`
  - 使用：`labelImg [图像目录] [预定义的类别文件]`
  - 预定义的类别文件为txt格式，每一行为一个类别的名称
- [`labelme`](https://github.com/wkentaro/labelme)

标注数据完成后，要将数据集划分为训练集和验证集（如8:2）。数据集较多情况下还可以划分一部分数据作为测试集。

通常情况下，要对训练集进行增广，通常情况下有如下几种方法：

- 裁剪
- 位移
- 缩放
- 翻转
- 颜色与亮度的调整

## 配置文件修改

模型训练使用的是`yolov8.yaml`配置文件，位于`ultralytics/cfg/models/v8`。

文件内容（下面的参数和相关，后续介绍）：

```yaml
# Ultralytics YOLO 🚀, AGPL-3.0 license
# YOLOv8 object detection model with P3-P5 outputs. For Usage examples see https://docs.ultralytics.com/tasks/detect

# Parameters
nc: 80 # number of classes
scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.33, 0.25, 1024] # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
  s: [0.33, 0.50, 1024] # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
  m: [0.67, 0.75, 768] # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
  l: [1.00, 1.00, 512] # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
  x: [1.00, 1.25, 512] # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs

# YOLOv8.0n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]] # 9

# YOLOv8.0n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 3, C2f, [512]] # 12

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 3, C2f, [256]] # 15 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]] # cat head P4
  - [-1, 3, C2f, [512]] # 18 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]] # cat head P5
  - [-1, 3, C2f, [1024]] # 21 (P5/32-large)

  - [[15, 18, 21], 1, Detect, [nc]] # Detect(P3, P4, P5)
```

## 模型训练

训练的代码非常简单：

```python
from ultralytics import YOLO
# swanlab 是察看训练过程的一些参数及损失函数相关内容，可省略，使用前要注册和安装
from swanlab.integration.ultralytics import add_swanlab_callback
import swanlab

def main():
    swanlab.init(project="your-project", experiment_name="your-project")
    # 这里是下载好的本地预训练模型文件，不写默认从网上下载，这里也可以用 yolov8.yaml 配置文件
    # 如果修改了模型参数，这里只能用 yolov8.yaml 配置文件
    model = YOLO("./yolov8n.pt")
    add_swanlab_callback(model)
    # 将下面的路径替换成你的绝对路径
    model.train(data="data.yaml", epochs=100, batch=16)

if __name__ == "__main__":
    main()
```

训练完成后，将在`runs/detect/train`目录下生成训练结果文件，其中生成模型文件路径在`weights`子目录下。

## 模型推理

一个简单的`web`页面用于测试推理的代码：

```python
import gradio as gr
from ultralytics import YOLO
from PIL import Image
import time
import cv2
import numpy as np

# 加载预训练的 YOLO 模型
model = YOLO('runs/detect/train/weights/best.pt')


def predict_image(image, conf_threshold=0.7, iou_threshold=0.5):
    # 使用模型进行推理
    results = model.predict(
        source=image,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=640, )

    # 提取结果
    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        return im


# 定义 Gradio 接口
demo = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(minimum=0, maximum=1, value=0.7, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.5, label="IoU threshold"),
    ],
    outputs=gr.Image(type="pil", label="Result"),
    title="stream",
    description="传一张包含图像来进行推理。",
)

# 启动 Gradio 应用
if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0')
```

`result`数据结构里面最主要的是`boxes`，对应识别出来的数据，其中重要的数据：

- cls: 类别
- conf：置信度
- xywh：X、Y坐标和宽度、高度值
- xywhn：X、Y坐标和宽度、高度的归一化值
- xyxy：左上和右下的X、Y坐标
- xyxyn：左上和右下的X、Y坐标的归一化值

## 实际应用中的问题

### 大图片而物体过小

这种场景适用于训练数据拍摄的是局部，以光伏板为例，一个光伏板的大小约2.5m*1.1m，而训练数据拍摄的损坏的局部部分，可能只有光伏板大小1/2甚至只有1/5大小。但是在无人机飞行过程中每次都是拍摄一片区域，包含2~4块光伏板，从而训练数据和实际数据不匹配，这样就导致了误判率比较高。

使用类似滑动窗口的方式，将一个大图片划分为多个小图片，示范代码如下：

```python
# 如果 yolo 识别图片最佳大小为 640 * 640（训练数据）
widget = 640
height = 640
x_stride = widget * 0.7 # 防止物体跨两个窗口
y_stride = height * 0.7
for y in range(0, image.shape[0], y_stride):
  for x in range(0, image.shape[1], x_stride):
    segment_image = image[y:y+height, x:x + widget]
    results = self.model.predict(source=segment_image, conf=conf, iou=iou, classes=predict_class)
    # 记录 boxes 信息，但是要去掉重复部分
```
