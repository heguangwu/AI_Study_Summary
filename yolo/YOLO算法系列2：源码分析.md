[TOC]

# `YOLO` 算法系列（2）源码分析

## `yolov8`架构

[`yolov8`架构](../img/yolov8架构图.jpg)

yolov8 的配置文件为： ```yolov8.yaml```


## `yolov8`配置

```yaml
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
  # 程序从 Detect 这个字段推断是什么任务，比如 Detect 或 Classify 或 Segment 或 Pose 等
  - [[15, 18, 21], 1, Detect, [nc]] # Detect(P3, P4, P5)
```

## 源码

### 入口点

入口文件：`ultralytics/models/yolo/model.py`

为省略，下面只列出核心代码，且以 `detect` 任务为例

```python
class YOLO(Model):
  def __init__(self, model: Union[str, Path] = "yolo11n.pt", task: Optional[str] = None, verbose: bool = False):
    # 调用父类的初始化方法
    super().__init__(model=model, task=task, verbose=verbose)

  def task_map(self) -> Dict[str, Dict[str, Any]]:
    """Map head to model, trainer, validator, and predictor classes."""
    return {
      "detect": {
          "model": DetectionModel,
          "trainer": yolo.detect.DetectionTrainer,
          "validator": yolo.detect.DetectionValidator,
          "predictor": yolo.detect.DetectionPredictor,
      },
    }

# Model部分

class Model(torch.nn.Module):
  def __init__(
        self,
        model: Union[str, Path, "Model"] = "yolo11n.pt",
        task: str = None,
        verbose: bool = False,
    ) -> None:
    # 判断输入的是模型配置文件还是模型文件
    if str(model).endswith((".yaml", ".yml")):
      self._new(model, task=task, verbose=verbose)
    else:
      self._load(model, task=task)

  def _new(self, cfg: str, task=None, model=None, verbose=False) -> None:
    # 这里才是根据yaml文件创建模型的地方
    self.model = (model or self._smart_load("model"))(cfg_dict, verbose=verbose and RANK == -1) 
  
  # 返回前面 task_map 中的  DetectionModel 对象（这里的 key 是 model）
  def _smart_load(self, key: str):
    return self.task_map[self.task][key]

```

接下来跟踪的 `DetectionModel` 代码如下：

```python
class DetectionModel(BaseModel):
  def __init__(self, cfg="yolo11n.yaml", ch=3, nc=None, verbose=True):
    self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)

```

在`parse_model`中，依次解析：

- nc: 类别数，通过`nc`获取
- activation：激活函数，通过`activation`获取
- scales：选择的模型是哪一个，通过`scale`获取，如无则取第一个模型，模型参数：
  - depth
  - width
  - max_channels

模型框架：

- 依次获取`backbone`和`head`
  - for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"])
