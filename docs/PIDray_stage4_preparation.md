# 阶段 4：PIDray 公共数据集验证准备记录

## 1. 阶段目标

本文档对应 `docs/2项目实施步骤.md` 中的阶段 4：PIDray 公共数据集验证。

阶段 4 的目标不是继续扩展模型，而是验证当前候选主线 `XR-Nano = YOLOv8n + PCN + EGI` 在公开 PIDray 数据集上的泛化能力，尤其关注 Easy / Hard / Hidden 三个官方测试子集。

## 2. 当前准备状态

当前阶段 4 已具备启动条件。

已完成准备：

1. PIDray 原始数据已存在于 `datasets/pidray/`。
2. 已建立 Ultralytics 兼容数据视图 `datasets/PIDray_yolo/`。
3. 已生成训练和测试 image list。
4. 已补齐 PIDray 普通 RGB 与 PCN+EGI 数据配置。
5. 已准备 YOLOv8n、YOLOv8s、XR-Nano 三个最小闭环模型的训练命令。
6. 已准备 Easy / Hard / Hidden / Overall 评估配置。
7. 已创建结果记录模板。

## 3. 数据目录

原始数据目录：

```text
datasets/pidray/
```

当前结构：

```text
datasets/pidray/train
datasets/pidray/easy
datasets/pidray/hard
datasets/pidray/hidden
datasets/pidray/labels/train
datasets/pidray/labels/easy
datasets/pidray/labels/hard
datasets/pidray/labels/hidden
```

Ultralytics 默认通过把图片路径中的 `images` 替换成 `labels` 来寻找标签，因此已经建立兼容视图：

```text
datasets/PIDray_yolo/images/train
datasets/PIDray_yolo/images/easy
datasets/PIDray_yolo/images/hard
datasets/PIDray_yolo/images/hidden
datasets/PIDray_yolo/labels/train
datasets/PIDray_yolo/labels/easy
datasets/PIDray_yolo/labels/hard
datasets/PIDray_yolo/labels/hidden
```

说明：这些目录是 junction 视图，指向原始 PIDray 数据，不复制图片和标签。

## 4. 数据规模

| Split | Images | Labels | Instances |
| --- | ---: | ---: | ---: |
| train | 29457 | 29457 | 39708 |
| easy | 9482 | 9482 | 9482 |
| hard | 3733 | 3733 | 8892 |
| hidden | 5005 | 5005 | 5008 |

标签已确认：

1. 格式为 YOLO txt。
2. 类别 ID 范围为 0-11。
3. 图片与标签数量一一对应。

## 5. 数据配置文件

训练配置：

```text
configs/datasets/PIDray.yaml
configs/datasets/PIDray_pcn_egi.yaml
```

普通 RGB 评估配置：

```text
configs/datasets/PIDray_test.yaml
configs/datasets/PIDray_easy.yaml
configs/datasets/PIDray_hard.yaml
configs/datasets/PIDray_hidden.yaml
```

PCN+EGI 评估配置：

```text
configs/datasets/PIDray_pcn_egi_test.yaml
configs/datasets/PIDray_pcn_egi_easy.yaml
configs/datasets/PIDray_pcn_egi_hard.yaml
configs/datasets/PIDray_pcn_egi_hidden.yaml
```

类别顺序与 SPXray 保持一致：

```text
Baton, Plier, Hammer, Powerbank, Scissors, Wrench, Gun, Bullet, Sprayer, HandCuffs, Knife, Lighter
```

## 6. 最小闭环模型

阶段 4 只先跑三个模型：

| 模型 | 目的 |
| --- | --- |
| YOLOv8n | 公共轻量 baseline |
| YOLOv8s | 强规模 baseline |
| XR-Nano PCN+EGI | 验证当前候选主线泛化 |

暂时不要跑 XR-Lite、P2-Lite、XR-Plus。只有当 XR-Nano 在 PIDray 上成立时，再考虑补 XR-Lite。

## 7. 训练命令

以下命令都在项目根目录执行。

如果你使用 `pg_moe` 环境，可以把命令开头的 `python` 替换成：

```powershell
conda run -n pg_moe python
```

### 7.1 YOLOv8n

```powershell
python -c "from ultralytics import YOLO; model=YOLO('G:/XR_Xray_Detection/yolov8n.pt'); model.train(data='G:/XR_Xray_Detection/configs/datasets/PIDray.yaml', epochs=300, imgsz=640, batch=16, device=0, optimizer='SGD', cos_lr=True, close_mosaic=10, seed=0, deterministic=True, pretrained=True, project='G:/XR_Xray_Detection/runs/train_pidray', name='yolov8n_pidray')"
```

### 7.2 YOLOv8s

```powershell
python -c "from ultralytics import YOLO; model=YOLO('G:/XR_Xray_Detection/yolov8s.pt'); model.train(data='G:/XR_Xray_Detection/configs/datasets/PIDray.yaml', epochs=300, imgsz=640, batch=16, device=0, optimizer='SGD', cos_lr=True, close_mosaic=10, seed=0, deterministic=True, pretrained=True, project='G:/XR_Xray_Detection/runs/train_pidray', name='yolov8s_pidray')"
```

### 7.3 XR-Nano PCN+EGI

```powershell
python -c "from ultralytics import YOLO; model=YOLO('G:/XR_Xray_Detection/yolov8n.pt'); model.train(data='G:/XR_Xray_Detection/configs/datasets/PIDray_pcn_egi.yaml', epochs=300, imgsz=640, batch=16, device=0, optimizer='SGD', cos_lr=True, close_mosaic=10, seed=0, deterministic=True, pretrained=True, project='G:/XR_Xray_Detection/runs/train_pidray', name='xr_nano_pidray')"
```

## 8. 评估命令

每个训练完成后，需要分别评估 overall、easy、hard、hidden。

### 8.1 YOLOv8n

```powershell
python -c "from ultralytics import YOLO; model=YOLO('G:/XR_Xray_Detection/runs/train_pidray/yolov8n_pidray/weights/best.pt'); model.val(data='G:/XR_Xray_Detection/configs/datasets/PIDray_test.yaml', imgsz=640, batch=16, device=0, project='G:/XR_Xray_Detection/runs/val_pidray', name='yolov8n_overall')"
python -c "from ultralytics import YOLO; model=YOLO('G:/XR_Xray_Detection/runs/train_pidray/yolov8n_pidray/weights/best.pt'); model.val(data='G:/XR_Xray_Detection/configs/datasets/PIDray_easy.yaml', imgsz=640, batch=16, device=0, project='G:/XR_Xray_Detection/runs/val_pidray', name='yolov8n_easy')"
python -c "from ultralytics import YOLO; model=YOLO('G:/XR_Xray_Detection/runs/train_pidray/yolov8n_pidray/weights/best.pt'); model.val(data='G:/XR_Xray_Detection/configs/datasets/PIDray_hard.yaml', imgsz=640, batch=16, device=0, project='G:/XR_Xray_Detection/runs/val_pidray', name='yolov8n_hard')"
python -c "from ultralytics import YOLO; model=YOLO('G:/XR_Xray_Detection/runs/train_pidray/yolov8n_pidray/weights/best.pt'); model.val(data='G:/XR_Xray_Detection/configs/datasets/PIDray_hidden.yaml', imgsz=640, batch=16, device=0, project='G:/XR_Xray_Detection/runs/val_pidray', name='yolov8n_hidden')"
```

### 8.2 YOLOv8s

```powershell
python -c "from ultralytics import YOLO; model=YOLO('G:/XR_Xray_Detection/runs/train_pidray/yolov8s_pidray/weights/best.pt'); model.val(data='G:/XR_Xray_Detection/configs/datasets/PIDray_test.yaml', imgsz=640, batch=16, device=0, project='G:/XR_Xray_Detection/runs/val_pidray', name='yolov8s_overall')"
python -c "from ultralytics import YOLO; model=YOLO('G:/XR_Xray_Detection/runs/train_pidray/yolov8s_pidray/weights/best.pt'); model.val(data='G:/XR_Xray_Detection/configs/datasets/PIDray_easy.yaml', imgsz=640, batch=16, device=0, project='G:/XR_Xray_Detection/runs/val_pidray', name='yolov8s_easy')"
python -c "from ultralytics import YOLO; model=YOLO('G:/XR_Xray_Detection/runs/train_pidray/yolov8s_pidray/weights/best.pt'); model.val(data='G:/XR_Xray_Detection/configs/datasets/PIDray_hard.yaml', imgsz=640, batch=16, device=0, project='G:/XR_Xray_Detection/runs/val_pidray', name='yolov8s_hard')"
python -c "from ultralytics import YOLO; model=YOLO('G:/XR_Xray_Detection/runs/train_pidray/yolov8s_pidray/weights/best.pt'); model.val(data='G:/XR_Xray_Detection/configs/datasets/PIDray_hidden.yaml', imgsz=640, batch=16, device=0, project='G:/XR_Xray_Detection/runs/val_pidray', name='yolov8s_hidden')"
```

### 8.3 XR-Nano PCN+EGI

```powershell
python -c "from ultralytics import YOLO; model=YOLO('G:/XR_Xray_Detection/runs/train_pidray/xr_nano_pidray/weights/best.pt'); model.val(data='G:/XR_Xray_Detection/configs/datasets/PIDray_pcn_egi_test.yaml', imgsz=640, batch=16, device=0, project='G:/XR_Xray_Detection/runs/val_pidray', name='xr_nano_overall')"
python -c "from ultralytics import YOLO; model=YOLO('G:/XR_Xray_Detection/runs/train_pidray/xr_nano_pidray/weights/best.pt'); model.val(data='G:/XR_Xray_Detection/configs/datasets/PIDray_pcn_egi_easy.yaml', imgsz=640, batch=16, device=0, project='G:/XR_Xray_Detection/runs/val_pidray', name='xr_nano_easy')"
python -c "from ultralytics import YOLO; model=YOLO('G:/XR_Xray_Detection/runs/train_pidray/xr_nano_pidray/weights/best.pt'); model.val(data='G:/XR_Xray_Detection/configs/datasets/PIDray_pcn_egi_hard.yaml', imgsz=640, batch=16, device=0, project='G:/XR_Xray_Detection/runs/val_pidray', name='xr_nano_hard')"
python -c "from ultralytics import YOLO; model=YOLO('G:/XR_Xray_Detection/runs/train_pidray/xr_nano_pidray/weights/best.pt'); model.val(data='G:/XR_Xray_Detection/configs/datasets/PIDray_pcn_egi_hidden.yaml', imgsz=640, batch=16, device=0, project='G:/XR_Xray_Detection/runs/val_pidray', name='xr_nano_hidden')"
```

## 9. 阶段 4 判断标准

优先看：

1. Hidden mAP50:95。
2. Hidden Recall。
3. Hard mAP50:95。
4. Overall mAP50:95。
5. FPS / Params / FLOPs。

判断边界：

| 结果 | 论文意义 |
| --- | --- |
| XR-Nano 在 Hidden / Hard 上明显提升 | 支撑遮挡/隐藏场景主张 |
| XR-Nano 只在 Easy 上提升 | 遮挡叙事变弱 |
| XR-Nano 完全不提升 | 当前论文主线只能先收缩到 SPXray |
| XR-Nano 提升但低于 YOLOv8s | 可写轻量模型收益，但不能写绝对最优 |

## 10. 当前不要做的事

1. 暂时不要训练 XR-Lite、P2-Lite、XR-Plus。
2. 暂时不要加入 BLT-MIX。
3. 暂时不要改 loss。
4. 暂时不要做蒸馏。
5. 暂时不要把 PIDray 结论写成已成立，必须等待训练和 Easy / Hard / Hidden 评估结果。
