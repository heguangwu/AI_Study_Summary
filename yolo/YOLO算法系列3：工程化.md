[TOC]

# `YOLO`算法系列（3）工程化

使用`YOLO`训练好之后的模型，可以使用`Python`很容易加载和运行，但如果要求较高性能就不太合适了，或者运行设备是一个边缘设备，GPU厂家和型号都不一致，甚至都无法安装`pytorch`等运行环境，由此可以看出，模型推理面临跨语言、跨平台、跨框架之间部署的问题，这时就需要一个开放模型格式及运行时来处理这些问题。

> `ONNX`（Open Neural Network Exchange，开放神经网络交换格式）是一种用于表示机器学习模型的开放格式标准，旨在解决不同深度学习框架之间模型格式不兼容的问题，实现模型在不同框架、平台间的无缝迁移和部署。
>
> `ONNX Runtime`：微软开发的跨平台高性能推理引擎，可直接加载并运行 `ONNX` 模型，支持 CPU、GPU、边缘设备等多种硬件，并通过优化（如算子融合、量化）提升推理效率，是 `ONNX` 生态的核心工具。
>
> 除了`ONNX Runtime`外，还有一个`TensorRT`推理引擎，是英伟达推出的主要用于在NVIDIA GPU上加速推理的框架。如果是英伟达的设备，最好使用这个框架
>

## 模型转换导出

```python
from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')
# 导出 onnx 并使用half设置为 fp16 精度（性能和精度均衡），输出文件为 best.onnx
model.export(format='onnx', half=True)
```

## 使用 `C++` 部署

到[官网](https://github.com/microsoft/onnxruntime)下载对应的`onnx runtime`版本，如下：

![`onnxruntime` 版本](../img/onnx_runtime_version.png)

解压即可使用。

[C++使用教程](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx)

新建一个`C++`工程，`CMakeLists.txt`如下：

```cmake
cmake_minimum_required(VERSION 3.10)
project(onnx_cpp)

set(CMAKE_CXX_STANDARD 17)

# 查找依赖
find_package(OpenCV REQUIRED)
set(ONNXRUNTIME_DIR "{onnxruntime解压后的文件夹}")

# 包含目录
include_directories(
    ${ONNXRUNTIME_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    include
)

# 源文件
set(SRC src/main.cpp)

# 主程序
add_executable(onnx_cpp  ${SRC})
target_link_libraries(onnx_cpp
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
)
```

`main.cpp`程序框架如下：

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"

int main() {
    // 1. 初始化ONNX Runtime环境
    Ort::Env env(nullptr) = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNX-Inference");

    // 2. 加载模型
    const char* model_path = "best.onnx";
    Ort::Session session = create_session(env, model_path);

    // 3. 获取模型信息
    std::vector<const char*> input_names = get_input_names(session;
    std::vector<const char*> output_names = get_output_names(session);
    auto input_shape = get_input_shape(session);
    auto output_shape = get_output_shape(session);
    
    // 4. 加载测试图像
    std::string image_path = "/home/bpz/yolo系列网课/yolov13_onnx_cpp/test_images/test.jpg";
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }
    cv::Size original_size = image.size();

    // 5. 图片预处理
    cv::Mat resized_image = preprocess_image(image, input_shape);
    std::vector<float> input_tensor = prepare_input_tensor(resized_image);

    // 6. 创建输入Tensor
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info, input_tensor.data(), input_tensor.size(),
        input_shape.data(), input_shape.size()));

    // 7. 运行推理
    std::vector<Ort::Value> output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names.data(), input_tensors.data(), input_names.size(),
        output_names.data(), output_names.size()
    );

    // 8. 获取输出数据
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> output_vector(output_data, output_data + output_size);

    // 9. 推理图片处理： todo

    // 10. 绘制结果
    for (const auto& det : detections) {
        //todo
    }

    // 11. 保存结果
    cv::imwrite("detection_result.jpg", image);

    // 12. 释放资源
    Ort::AllocatorWithDefaultOptions allocator;
    for (auto name : input_names) {
        allocator.Free(const_cast<void*>(static_cast<const void*>(name)));
    }
    for (auto name : output_names) {
        allocator.Free(const_cast<void*>(static_cast<const void*>(name)));
    }

    return 0;
}
```

同样使用`onnx`，相比`Python`程序，`C++`性能可以提升50%以上。
