# P1 阶段 README：SPXray 结果闭环验收与下一步执行清单

## 1. 本阶段参考文件

本次阶段 1 全权参考以下两个文件：

1. `docs/完整版试验设计方案.md`
2. `docs/2项目实施步骤.md`

需要先说明一个命名差异：`docs/2项目实施步骤.md` 中的“阶段 1”指的是 **SPXray 结果闭环**；同一文件后面的“P1：接着做”指的是下一轮三 seed、PIDray、OPIXray、鲁棒性和可视化等工作。本 README 按“阶段 1：SPXray 结果闭环”来验收，同时把后续 P1 要做的事情列清楚。

## 2. 阶段 1 的目标

阶段 1 的核心目标不是继续增加新模块，而是把 SPXray 上已经完成的 baseline、输入侧消融和结构增强实验整理成论文级别的统一结果闭环。

按照 `docs/2项目实施步骤.md` 的要求，本阶段必须完成：

1. 统一主结果表。
2. 每类别 AP 表。
3. AP-small / AP-medium / AP-large。
4. AP-thin / Recall-thin。
5. overlap AP / overlap Recall。
6. 统一速度与复杂度结果。
7. 给出当前候选主线模型判断。

## 3. 本次已经完成和验收的内容

### 3.1 已完成统一主结果表

文件位置：

```text
runs/summary/spxray_main_results.csv
```

该表已经覆盖 8 个模型：

1. YOLOv8n
2. YOLOv8s
3. A1-PCN
4. A2-EGI
5. XR-Nano
6. XR-Lite
7. P2-Lite
8. XR-Plus

主表已经包含以下关键列：

```text
Params_M
FLOPs_G
FPS
Precision
Recall
mAP50
mAP50_95
AP_small
R_small
AP_medium
R_medium
AP_large
R_large
AP_thin
R_thin
AP_overlap_0p3
R_overlap_0p3
AP_overlap_0p5
R_overlap_0p5
```

这已经满足阶段 1 对“统一主表”的要求。

### 3.2 已完成每类别 AP 表

文件位置：

```text
runs/summary/spxray_per_class_ap.csv
```

当前表共有 96 行，对应：

```text
8 个模型 × 12 个 SPXray 类别 = 96 行
```

每一行包含类别级 precision、recall、F1、AP50、AP75、AP50:95，可用于论文中的 per-class AP 分析。

### 3.3 已完成难样本子集评估

文件目录：

```text
runs/summary/subset_metrics/
```

当前目录共有 48 个 JSON 文件，对应：

```text
8 个模型 × 6 个子集 = 48 个结果文件
```

已覆盖的子集包括：

1. small
2. medium
3. large
4. thin
5. overlap_0p3
6. overlap_0p5

这部分满足阶段 1 对 AP-small、AP-medium、AP-large、AP-thin、overlap AP 的要求。

### 3.4 已完成 SPXray 数据统计表

文件位置：

```text
runs/summary/spxray_dataset_stats.csv
runs/summary/spxray_dataset_stats/
```

当前统计已经覆盖 train、val、test 三个 split，并包含：

1. 图像数量。
2. 实例数量。
3. small / medium / large 数量与比例。
4. thin object 数量与比例。
5. overlap@0.3 数量与比例。
6. overlap@0.5 数量与比例。

注意：SPXray 当前没有 hidden 标注，因此论文和报告中应写 `overlap`，不要写成 `hidden`。

### 3.5 已完成模型注册与主线冻结

文件位置：

```text
runs/summary/model_registry.csv
docs/current_mainline.md
```

`model_registry.csv` 已经固定每个模型对应的：

1. 模型名称。
2. 角色定位。
3. 模型配置。
4. 数据配置。
5. 训练目录。
6. 权重路径。
7. 预测标签路径。
8. PCN / EGI / EMA-Lite / AIFI-Lite / P2-Lite 开关。

当前候选主线仍然冻结为：

```text
XR-Nano = YOLOv8n + PCN + EGI
```

但这里必须严格表述：在当前重新验证结果中，A2-EGI 的 mAP50:95 略高于 XR-Nano，因此不能写“XR-Nano 是绝对 mAP 最优模型”。XR-Nano 被保留为候选主线，是因为它在轻量化、AP-thin、PCN+EGI 方法完整性和后续论文主张上更平衡。

### 3.6 已完成速度与复杂度汇总

阶段 1 主表已经包含 Params、FLOPs 和 FPS。已有速度与复杂度相关文件包括：

```text
runs/summary/baseline_speed/model_speed_complexity.csv
runs/summary/a1_speed/model_speed_complexity.csv
runs/summary/a2_speed/model_speed_complexity.csv
```

后续论文正式速度实验还需要补 ONNX / TensorRT / latency mean±std，但这属于部署速度实验，不属于本阶段必须项。

## 4. 阶段 1 结论

阶段 1 的 SPXray 结果闭环已经完成，可以进入下一轮实验。

当前最稳妥的论文主线表述是：

```text
面向伪彩色 X 射线安检图像的边缘结构引导输入增强方法
```

当前候选主模型是：

```text
XR-Nano = YOLOv8n + PCN + EGI
```

XR-Lite、P2-Lite、XR-Plus 暂时不提升为最终主模型，只作为结构增强未带来净收益或代价较高的消融证据保留。

## 5. 本阶段没有做的事情

本阶段没有启动新的 300 epoch 长训练。

原因是：按照 `docs/2项目实施步骤.md`，阶段 1 的任务是整理和闭环 SPXray 已有结果；下一步三 seed 稳定性实验属于后续 P1 / 阶段 2 工作，训练成本较高，需要单独执行和记录。

## 6. 你接下来需要做什么

### 6.1 优先做 YOLOv8n 与 XR-Nano 的三 seed 稳定性实验

按照 `docs/2项目实施步骤.md` 和 `docs/实验协议.md`，正式论文结果不能只依赖单 seed。下一步只需要对核心模型做三 seed，不要把所有模型都重跑三 seed。

需要跑的模型：

| 模型 | seed |
| --- | --- |
| YOLOv8n | 0, 1, 2 |
| XR-Nano PCN+EGI | 0, 1, 2 |

当前已有 seed=0 的历史结果，可以作为 pilot 参考。但如果要做最干净的论文结果，建议在同一个新目录下重新跑 0、1、2 三个 seed，避免历史训练目录、数据配置路径和当前项目本地路径不完全一致。

建议新目录：

```text
runs/p1_seed_stability/
```

### 6.2 建议执行命令

在项目根目录执行。下面命令默认使用当前 Python 环境；如果你平时使用 `conda run -n pg_moe python`，把命令开头的 `python` 替换成 `conda run -n pg_moe python`。

YOLOv8n seed 0：

```powershell
python -c "from ultralytics import YOLO; model=YOLO('G:/XR_Xray_Detection/runs/train_spxray/yolov8n_spxray/weights/best.pt'); model.train(data='G:/XR_Xray_Detection/configs/datasets/SPXray.yaml', epochs=200, imgsz=640, batch=16, device=0, optimizer='SGD', cos_lr=True, close_mosaic=10, seed=0, deterministic=True, pretrained=True, project='G:/XR_Xray_Detection/runs/p1_seed_stability', name='yolov8n_seed0')"
```

YOLOv8n seed 1：

```powershell
python -c "from ultralytics import YOLO; model=YOLO('G:/XR_Xray_Detection/runs/train_spxray/yolov8n_spxray/weights/best.pt'); model.train(data='G:/XR_Xray_Detection/configs/datasets/SPXray.yaml', epochs=200, imgsz=640, batch=16, device=0, optimizer='SGD', cos_lr=True, close_mosaic=10, seed=1, deterministic=True, pretrained=True, project='G:/XR_Xray_Detection/runs/p1_seed_stability', name='yolov8n_seed1')"
```

YOLOv8n seed 2：

```powershell
python -c "from ultralytics import YOLO; model=YOLO('G:/XR_Xray_Detection/runs/train_spxray/yolov8n_spxray/weights/best.pt'); model.train(data='G:/XR_Xray_Detection/configs/datasets/SPXray.yaml', epochs=200, imgsz=640, batch=16, device=0, optimizer='SGD', cos_lr=True, close_mosaic=10, seed=2, deterministic=True, pretrained=True, project='G:/XR_Xray_Detection/runs/p1_seed_stability', name='yolov8n_seed2')"
```

XR-Nano seed 0：

```powershell
python -c "from ultralytics import YOLO; model=YOLO('G:/XR_Xray_Detection/runs/train_stage2/yolov8n_pcn_egi/weights/best.pt'); model.train(data='G:/XR_Xray_Detection/configs/datasets/SPXray_pcn_egi.yaml', epochs=200, imgsz=640, batch=16, device=0, optimizer='SGD', cos_lr=True, close_mosaic=10, seed=0, deterministic=True, pretrained=True, project='G:/XR_Xray_Detection/runs/p1_seed_stability', name='xr_nano_seed0')"
```

XR-Nano seed 1：

```powershell
python -c "from ultralytics import YOLO; model=YOLO('G:/XR_Xray_Detection/runs/train_stage2/yolov8n_pcn_egi/weights/best.pt'); model.train(data='G:/XR_Xray_Detection/configs/datasets/SPXray_pcn_egi.yaml', epochs=200, imgsz=640, batch=16, device=0, optimizer='SGD', cos_lr=True, close_mosaic=10, seed=1, deterministic=True, pretrained=True, project='G:/XR_Xray_Detection/runs/p1_seed_stability', name='xr_nano_seed1')"
```

XR-Nano seed 2：

```powershell
python -c "from ultralytics import YOLO; model=YOLO('G:/XR_Xray_Detection/runs/train_stage2/yolov8n_pcn_egi/weights/best.pt'); model.train(data='G:/XR_Xray_Detection/configs/datasets/SPXray_pcn_egi.yaml', epochs=200, imgsz=640, batch=16, device=0, optimizer='SGD', cos_lr=True, close_mosaic=10, seed=2, deterministic=True, pretrained=True, project='G:/XR_Xray_Detection/runs/p1_seed_stability', name='xr_nano_seed2')"
```

### 6.3 三 seed 完成后需要汇总的表

三 seed 完成后，需要生成以下表：

| Model | mAP50:95 | AP-small | AP-thin | FPS |
| --- | --- | --- | --- | --- |
| YOLOv8n | mean ± std | mean ± std | mean ± std | mean |
| XR-Nano | mean ± std | mean ± std | mean ± std | mean |

判断规则：

1. 如果提升小于标准差，不能写“显著提升”。
2. 如果总体 mAP 提升不大，但 AP-thin 稳定提升，可以把论文主张收窄到边缘结构、细长目标和难样本。
3. 如果 XR-Nano 三 seed 不稳定，当前结果更适合开题或阶段性报告，不适合直接投稿。

### 6.4 三 seed 后再做公共数据集验证

三 seed 稳定性完成后，下一步再做：

1. PIDray 最小公共数据集验证。
2. OPIXray 遮挡与细长目标验证。
3. 鲁棒性实验。
4. YOLOv8n vs XR-Nano 可视化对比。
5. 失败案例分析。

BLT-MIX 暂时不要提前加入主线。它应当放在 XR-Nano 主线、三 seed 稳定性和最小公共数据集验证之后，作为独立的数据增强消融实验。

## 7. 写论文时必须注意的表述边界

1. 不要说 XR-Nano 是“所有指标绝对最优”。
2. 可以说 XR-Nano 是当前 SPXray 上综合更平衡的候选主线。
3. SPXray 没有 hidden 标注时，只写 overlap，不写 hidden。
4. XR-Lite、P2-Lite、XR-Plus 当前只能作为消融证据，不建议写成最终主模型。
5. 三 seed 完成前，不要写“显著提升”。
6. 公共数据集验证完成前，不要写“泛化能力已充分证明”。

## 8. 当前阶段状态

阶段 1 状态：

```text
已完成
```

下一步优先级：

```text
YOLOv8n 与 XR-Nano 三 seed 稳定性实验
```
