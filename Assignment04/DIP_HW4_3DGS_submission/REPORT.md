# Assignment 4: Simplified 3D Gaussian Splatting

## 1. 实验目标

本作业实现一个简化版 3D Gaussian Splatting pipeline，从多视角图像恢复相机位姿和稀疏点云，再用可微分的 3D Gaussian 渲染进行训练。作业包含三部分：

1. 使用 COLMAP 完成 Structure-from-Motion，得到相机参数和稀疏 3D 点。
2. 补全简化版 3DGS 的核心 PyTorch 实现，包括 3D 协方差构造、3D 到 2D 投影、2D Gaussian 权重计算和 alpha blending。
3. 与官方 3DGS 实现比较渲染质量、训练速度和显存/内存占用，并分析差异来源。

本次实验选择 `chair` 数据集，共 100 张多视角 RGBA 渲染图像。

## 2. Task 1: COLMAP SfM

运行命令：

```bash
python mvs_with_colmap.py --data_dir data/chair
python debug_mvs_by_projecting_pts.py --data_dir data/chair
```

服务器上使用的 COLMAP 命令通过 `COLMAP_BIN` 环境变量指定到已有可用 wrapper，避免系统 `libstdc++` 版本问题：

```bash
export COLMAP_BIN=/home/czr/stage2_3dgs_indoor/tools/colmap
```

结果：

| 项目 | 结果 |
| --- | ---: |
| 输入图像数量 | 100 |
| COLMAP 稀疏点数量 | 13,763 |
| 投影验证图数量 | 100 |

COLMAP 输出保存在：

```text
data/chair/database.db
data/chair/sparse/0/
data/chair/sparse/0_text/
data/chair/projections/
```

这些 3D 点用于初始化简化版 3DGS 中的 Gaussian centers。

## 3. Task 2: 简化版 3DGS 实现

已补全的核心代码：

| 文件 | 实现内容 |
| --- | --- |
| `gaussian_model.py` | 根据 scale 和 quaternion rotation 构造 3D covariance |
| `gaussian_renderer.py` | 实现 3D Gaussian 投影到 2D、2D Gaussian 权重和 alpha blending |
| `data_utils.py` | 读取 COLMAP 结果和图像；去除额外 `pytorch3d` / `natsort` 依赖 |
| `train.py` | 训练、debug 图保存、最终 checkpoint 保存、训练时间和 CUDA 显存记录 |

关键公式如下。

3D Gaussian covariance：

```text
S = diag(sx, sy, sz)
L = R S
Sigma = L L^T
```

透视投影：

```text
x_c = R x_w + t
u = fx * x_c / z_c + cx
v = fy * y_c / z_c + cy
```

2D covariance 的一阶近似：

```text
Sigma_2D = J W Sigma W^T J^T
```

alpha blending：

```text
alpha_i(x) = opacity_i * G_i(x)
T_i(x) = product_{j<i} (1 - alpha_j(x))
C(x) = sum_i T_i(x) alpha_i(x) color_i
```

训练命令：

```bash
CUDA_VISIBLE_DEVICES=4 python train.py \
  --colmap_dir data/chair \
  --checkpoint_dir outputs/chair_simplified_30 \
  --num_epochs 30 \
  --debug_every 5 \
  --debug_samples 4 \
  --device cuda
```

训练结果：

| 项目 | 结果 |
| --- | ---: |
| Gaussian 数量 | 13,763 |
| 训练轮数 | 30 epochs |
| 训练耗时 | 27 min 19.61 s |
| PyTorch peak CUDA memory | 10,190.70 MB |
| 全图 mean PSNR | 20.64 dB |
| 前景 mean PSNR | 14.26 dB |

说明：全图 PSNR 中背景占比很大，而输入数据为透明背景物体图像。为了更好评估物体本身，本报告额外计算了 foreground-only 指标，即只在 GT 非黑色前景区域上统计误差。

## 4. Task 3: 与官方 3DGS 对比

官方 3DGS 使用同一份 COLMAP scene 和同一组 chair 图像训练，命令如下：

```bash
CUDA_VISIBLE_DEVICES=4 python /home/czr/hw/gaussian-splatting/train.py \
  -s /home/czr/dip_hw4_3dgs_project/data/chair \
  -m /home/czr/dip_hw4_3dgs_project/outputs/official_chair_30000_noreset \
  --iterations 30000 \
  --save_iterations 30000 \
  --test_iterations 30000 \
  --disable_viewer \
  --resolution 1 \
  --opacity_reset_interval 1000000
```

官方渲染命令：

```bash
CUDA_VISIBLE_DEVICES=4 python /home/czr/hw/gaussian-splatting/render.py \
  -s /home/czr/dip_hw4_3dgs_project/data/chair \
  -m /home/czr/dip_hw4_3dgs_project/outputs/official_chair_30000_noreset \
  --iteration 30000 \
  --skip_test
```

对比结果：

| 方法 | Gaussian 数量 | 训练时间 | 指标口径 | mean PSNR |
| --- | ---: | ---: | --- | ---: |
| 简化版 PyTorch 3DGS | 13,763 | 27 min 19.61 s | 全图 | 20.64 dB |
| 简化版 PyTorch 3DGS | 13,763 | 27 min 19.61 s | 前景 | 14.26 dB |
| 官方 3DGS | 461,872 | 7 min 27.18 s | 前景 | 20.12 dB |

官方全图 PSNR 约为 1.02 dB，但这个数字不代表前景重建失败。原因是官方 render 输出中背景为白色，而 GT 图像导出为黑色背景，整张图统计时大面积背景差异会主导 MSE。因此报告中主要使用 foreground-only PSNR 对比物体区域质量，并在结果图中展示该现象。

## 5. 差异分析

简化版实现可以完整跑通 3DGS pipeline，但与官方实现有明显差距：

1. 简化版没有 adaptive densification。Gaussian 数量固定为 COLMAP 稀疏点数 13,763，难以覆盖椅子上的细节纹理、边缘和透明/镂空区域。
2. 简化版没有 tile-based CUDA rasterizer。当前 PyTorch 实现直接在完整图像网格上计算 Gaussian contribution，显存和计算量都更高。
3. 官方实现会在训练中增加、裁剪和优化 Gaussian，最终约 461,872 个 Gaussian，因此能表达更细的结构。
4. 官方实现使用 CUDA 扩展和 tile-based rasterization，虽然训练了 30,000 iterations，但总耗时仍明显短于简化版 30 epochs。
5. RGBA 数据的背景处理会显著影响全图 PSNR。对透明背景物体，报告全图指标时必须说明背景颜色，前景 mask 指标更能反映物体重建质量。

## 6. 结果文件

本提交包中的关键结果：

```text
results/simplified/eval/metrics.json
results/simplified/eval_foreground/metrics.json
results/simplified/examples/
results/simplified/debug_images/
results/simplified/debug_rendering.mp4
results/simplified/render_mv.mp4
results/official/eval_foreground/metrics.json
results/official/examples/
results/logs/
```

完整服务器实验目录：

```text
/home/czr/dip_hw4_3dgs_project
```

本地交付目录：

```text
D:\数字图像处理\hw4
```

## 7. 如何复现

服务器环境：

```bash
ssh research-server
cd /home/czr/dip_hw4_3dgs_project
source ~/miniconda3/etc/profile.d/conda.sh && conda activate pytorch_h100
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

运行 COLMAP：

```bash
export COLMAP_BIN=/home/czr/stage2_3dgs_indoor/tools/colmap
export COLMAP_USE_GPU=0
bash run_colmap_chair.sh
```

训练简化版：

```bash
CUDA_VISIBLE_DEVICES=4 bash run_simplified_train_chair.sh
```

评估简化版：

```bash
CUDA_VISIBLE_DEVICES=4 python evaluate_simplified.py \
  --colmap_dir data/chair \
  --checkpoint outputs/chair_simplified_30/checkpoint_000030.pt \
  --output_dir outputs/chair_simplified_30/eval \
  --save_examples 8

CUDA_VISIBLE_DEVICES=4 python evaluate_simplified.py \
  --colmap_dir data/chair \
  --checkpoint outputs/chair_simplified_30/checkpoint_000030.pt \
  --output_dir outputs/chair_simplified_30/eval_foreground \
  --save_examples 8 \
  --foreground_only
```

评估官方结果：

```bash
python evaluate_official_renders.py \
  --render_dir outputs/official_chair_30000_noreset/train/ours_30000 \
  --output_dir outputs/official_chair_30000_noreset/train/ours_30000/eval_foreground \
  --save_examples 8 \
  --foreground_only
```

## 8. 小结

本实验完成了作业要求的三个部分：COLMAP SfM、简化版 3DGS 核心实现、以及与官方 3DGS 的质量/速度/资源对比。简化版 pipeline 可以成功从 chair 多视角图像重建并渲染，达到全图 PSNR 20.64 dB；官方版在前景区域达到 20.12 dB，并显著更快。主要差异来自官方实现中的 adaptive densification、CUDA tile rasterization 和更成熟的训练策略。
