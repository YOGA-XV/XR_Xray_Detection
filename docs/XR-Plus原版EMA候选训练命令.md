# XR-Plus 原版 EMA 候选训练命令

本文档用于准备两个新的 XR-Plus 候选结构，目标是验证：在 `nano+原版 EMA` 和 `nano+原版 EMA + AIFI-Lite` 表现较好的基础上，追加 P2-Lite 分支后是否能进一步提升小目标或整体检测性能。

## 1. 候选结构

| 候选名 | 模型配置 | 初始化权重 | 结构说明 |
| --- | --- | --- | --- |
| XR-Plus-Original-EMA | `ultralytics/cfg/models/v8/yolov8n-xr-plus-original-ema.yaml` | `runs/train_attention_ablation/nano_original_ema/weights/best.pt` | XR-Nano + 原版 EMA + P2-Lite |
| XR-Plus-Original-EMA-AIFI-Lite | `ultralytics/cfg/models/v8/yolov8n-xr-plus-original-ema-aifi-lite.yaml` | `runs/train_attention_ablation/nano_original_ema_aifi_lite/weights/best.pt` | XR-Nano + 原版 EMA + AIFI-Lite + P2-Lite |

说明：

1. 两个模型都使用 `configs/datasets/SPXray_pcn_egi.yaml`，即 `PCN + EGI` 四通道输入。
2. 两个模型都不是从零训练，而是用对应 `nano+...` 已训练权重做部分初始化。
3. 因为 Plus 结构新增 P2 检测分支，P2 分支相关层会随机初始化，这是正常现象。
4. 本实验使用 `imgsz=768`，因此结果属于高分辨率 Plus 候选实验，不能和 `imgsz=640` 的结构消融直接混为同一变量。

## 2. 统一训练参数

| 参数 | 设置 |
| --- | --- |
| `epochs` | `300` |
| `imgsz` | `768` |
| `batch` | `16` |
| `workers` | `4` |
| `optimizer` | `SGD` |
| `lr0` | `0.0005` |
| `lrf` | `0.01` |
| `cos_lr` | `True` |
| `close_mosaic` | `20` |
| `seed` | `0` |
| `deterministic` | `False` |
| `patience` | `0` |
| 输出目录 | `runs/train_plus_original_ema_candidates` |

## 3. 一键顺序训练

推荐使用下面的批处理脚本。它会按顺序训练两个候选模型，并自动跳过已经完整跑完 300 轮的模型。

```bat
cd /d G:\XR_Xray_Detection
scripts\run_plus_original_ema_candidates.bat
```

也可以直接运行 Python 调度器：

```powershell
conda run --no-capture-output -n pg_moe python scripts\run_plus_original_ema_candidates.py
```

如果只想检查计划，不启动训练：

```powershell
conda run -n pg_moe python scripts\run_plus_original_ema_candidates.py --dry-run
```

如果需要强制重跑已完成模型：

```powershell
conda run --no-capture-output -n pg_moe python scripts\run_plus_original_ema_candidates.py --rerun-completed
```

## 4. 单独训练命令

### 4.1 XR-Plus-Original-EMA

```powershell
conda run --no-capture-output -n pg_moe python -c "from ultralytics import YOLO; model=YOLO(r'G:/XR_Xray_Detection/ultralytics/cfg/models/v8/yolov8n-xr-plus-original-ema.yaml'); model.train(data=r'G:/XR_Xray_Detection/configs/datasets/SPXray_pcn_egi.yaml', epochs=300, imgsz=768, batch=16, device=0, workers=4, optimizer='SGD', lr0=0.0005, lrf=0.01, cos_lr=True, close_mosaic=20, seed=0, deterministic=False, pretrained=r'G:/XR_Xray_Detection/runs/train_attention_ablation/nano_original_ema/weights/best.pt', patience=0, project=r'G:/XR_Xray_Detection/runs/train_plus_original_ema_candidates', name='xr_plus_original_ema_img768_lr0005_cm20')"
```

### 4.2 XR-Plus-Original-EMA-AIFI-Lite

```powershell
conda run --no-capture-output -n pg_moe python -c "from ultralytics import YOLO; model=YOLO(r'G:/XR_Xray_Detection/ultralytics/cfg/models/v8/yolov8n-xr-plus-original-ema-aifi-lite.yaml'); model.train(data=r'G:/XR_Xray_Detection/configs/datasets/SPXray_pcn_egi.yaml', epochs=300, imgsz=768, batch=16, device=0, workers=4, optimizer='SGD', lr0=0.0005, lrf=0.01, cos_lr=True, close_mosaic=20, seed=0, deterministic=False, pretrained=r'G:/XR_Xray_Detection/runs/train_attention_ablation/nano_original_ema_aifi_lite/weights/best.pt', patience=0, project=r'G:/XR_Xray_Detection/runs/train_plus_original_ema_candidates', name='xr_plus_original_ema_aifi_lite_img768_lr0005_cm20')"
```

## 5. 结果判断

训练完成后，优先比较：

1. `mAP50:95` 是否超过当前 XR-Lite 和原 XR-Plus。
2. `AP-small` 是否明显高于原 XR-Plus。
3. `AP-thin` 是否不低于 `nano+原版 EMA + AIFI-Lite`。
4. 若只提升 AP-small 但整体 mAP50:95 明显下降，则仍作为 P2 分支负代价消融，不建议提升为最终结构。
