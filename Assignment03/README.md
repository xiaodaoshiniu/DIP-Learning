# Assignment 03 - Bundle Adjustment

本次作业对应课程仓库中的 `03_BundleAdjustment`，主要完成两部分内容：

1. 使用 PyTorch 从零实现 Bundle Adjustment，联合优化相机参数与三维点云。
2. 使用 COLMAP 对给定的 50 张渲染图像进行稀疏与稠密三维重建。

## 目录结构

```text
Assignment03/
├─ bundle_adjustment.py          # PyTorch Bundle Adjustment 主程序
├─ visualize_data.py             # 二维观测点可视化脚本
├─ requirements.txt
├─ run_task1_full.sh             # 服务器完整 BA 运行脚本
├─ run_task1_debug.sh            # 小规模调试脚本
├─ run_colmap.sh                 # COLMAP 稀疏 + 稠密重建
├─ run_colmap_sparse.sh          # 仅运行 COLMAP 稀疏重建
├─ data/
│  ├─ images/                    # 50 张输入图像
│  ├─ vis/                       # 输入观测点可视化
│  ├─ points2d.npz
│  └─ points3d_colors.npy
└─ results/
   ├─ task1_ba_full/             # PyTorch BA 结果
   └─ colmap/                    # 精简后的 COLMAP 结果
```

## Task 1: PyTorch Bundle Adjustment

### 方法

相机内参使用共享焦距 `f`，主点固定在图像中心。每个视角优化一组欧拉角旋转与平移向量，同时优化全部 20000 个三维点。投影模型为：

```text
[Xc, Yc, Zc] = R @ P + T
u = -f * Xc / Zc + cx
v =  f * Yc / Zc + cy
```

优化目标为可见二维观测点与投影点之间的鲁棒重投影误差。实现中使用 Adam 优化器，并用 mini-batch 采样可见观测以降低显存占用。

### 运行命令

服务器完整运行命令如下：

```bash
python bundle_adjustment.py \
  --data-dir data \
  --output-dir results/task1_ba_full \
  --device cuda \
  --iters 3000 \
  --batch-size 262144 \
  --log-every 50
```

### 结果

| 指标 | 数值 |
| --- | ---: |
| 图像数量 | 50 |
| 优化三维点数量 | 20000 |
| 可见二维观测数量 | 805089 |
| 迭代次数 | 3000 |
| 最终焦距 | 3357.17 |
| 最终训练损失 | 8.92 px |
| 最终 RMSE | 8.96 px |
| 最终平均 L2 重投影误差 | 8.78 px |

损失曲线：

<img src="./results/task1_ba_full/loss_curve.png" alt="Bundle Adjustment loss curve" width="800">

优化后的彩色三维点云预览：

<img src="./results/task1_ba_full/point_cloud_preview.png" alt="Bundle Adjustment point cloud preview" width="800">

主要输出文件：

- `results/task1_ba_full/reconstruction.obj`
- `results/task1_ba_full/camera_params.npz`
- `results/task1_ba_full/training_log.csv`
- `results/task1_ba_full/metrics.json`

## Task 2: COLMAP Reconstruction

### 运行命令

完整的 COLMAP 流程包括特征提取、特征匹配、稀疏重建、图像去畸变、PatchMatch Stereo 与 Stereo Fusion：

```bash
bash run_colmap.sh
```

如果机器不支持稠密重建，也可以只运行稀疏重建：

```bash
bash run_colmap_sparse.sh data
```

### 结果

本次完整 COLMAP 流程已运行完成。为了避免提交 1GB 以上的中间缓存，仓库中仅保留关键结果文件：

| 结果 | 数值 |
| --- | ---: |
| 注册图像数量 | 50 |
| 相机数量 | 1 |
| 稀疏点数量 | 1703 |
| 稠密融合点数量 | 111824 |

COLMAP 稠密重建预览：

<img src="./results/colmap/dense/colmap_dense_preview.png" alt="COLMAP dense reconstruction preview" width="900">

保留的 COLMAP 结果文件：

- `results/colmap/sparse/0/cameras.bin`
- `results/colmap/sparse/0/images.bin`
- `results/colmap/sparse/0/points3D.bin`
- `results/colmap/dense/fused.ply`

## 输入数据可视化

二维观测点示例：

<img src="./data/vis/view_025_overlay.png" alt="2D observations overlay" width="500">

作业数据示意图：

<img src="./pics/data_overview.png" alt="Assignment data overview" width="800">

## 说明

- `results/colmap/dense/fused.ply` 是 COLMAP 稠密融合后的点云文件。
- `results/task1_ba_full/reconstruction.obj` 是 PyTorch BA 优化得到的彩色三维点云。
- COLMAP 的 `database.db`、depth maps、normal maps 等大体积中间文件未提交，避免仓库过大；保留的 sparse model 与 fused point cloud 已足够复查重建结果。
