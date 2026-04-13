# DIP-Learning

数字图像处理课程作业仓库，当前按作业编号进行整理，便于查看与提交。

## 作业导航

- [第一次作业：Image Warping](./Assignment01/README.md)
- [第二次作业：Poisson Image Editing 与 Pix2Pix](./Assignment02/README.md)

## 目录结构

```text
DIP-Learning/
├─ Assignment01/
│  ├─ README.md
│  ├─ requirements.txt
│  ├─ run_global_transform.py
│  ├─ run_point_transform.py
│  └─ test_pics/
└─ Assignment02/
   ├─ README.md
   ├─ Poisson/
   └─ Pix2Pix/
```

## 说明

- `Assignment01` 对应第一次作业，内容为图像几何变换与基于控制点的图像形变。
- `Assignment02` 对应第二次作业，内容为 Poisson 图像融合与 Pix2Pix 图像到图像翻译。
- 第二次作业中仅保留了与提交直接相关的代码、结果图和中文说明文档，效果较差的 Pix2Pix 结果图未放入提交目录。

## 作业预览

### 第一次作业：Image Warping

全局几何变换示例：

<img src="./Assignment01/test_pics/global_transform.png" alt="Assignment 1 Global Transform" width="800">

控制点形变示例：

<img src="./Assignment01/test_pics/point_transform.png" alt="Assignment 1 Point Transform" width="800">

### 第二次作业：Poisson Image Editing 与 Pix2Pix

Poisson 融合效果：

<img src="./Assignment02/Poisson/poisson_water_comparison.png" alt="Assignment 2 Poisson Result" width="800">

Pix2Pix 结果示例：

<img src="./Assignment02/Pix2Pix/figures/pix2pix_result_1.png" alt="Assignment 2 Pix2Pix Result 1" width="800">
