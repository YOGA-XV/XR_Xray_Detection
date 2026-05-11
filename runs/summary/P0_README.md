# P0 汇总包

## 指标口径

P0 表格优先使用 `best.pt` 重新验证得到的结果，而不是只读取训练过程 `results.csv` 中的 best 行。

主对比表为：

```text
runs/summary/spxray_main_results.csv
```

该表包含：

- Params 和 FLOPs。
- 端到端 FPS。
- Precision、Recall、mAP50、mAP50:95。
- small、medium、large、thin、overlap@0.3、overlap@0.5 子集上的 AP / Recall。

## 主线口径

当前冻结的候选主线为：

```text
XR-Nano = YOLOv8n + PCN + EGI
```

在当前重新验证表中，A2-EGI 的 mAP50:95 略高于 XR-Nano，因此不能把 XR-Nano 写成“绝对 mAP 最优模型”。XR-Nano 仍保留为候选主线，是因为它在论文叙事上更均衡：PCN+EGI 设计逻辑完整，AP-thin 有提升，模型代价轻量，并且优于后续结构增强版本的综合表现。

## 文件清单

```text
model_registry.csv
spxray_main_results.csv
spxray_per_class_ap.csv
spxray_dataset_stats.csv
spxray_dataset_stats/
subset_metrics/
baseline_speed/
a1_speed/
a2_speed/
```

## 下一阶段

下一步应从以下模型的 3 seed 稳定性实验开始：

```text
YOLOv8n
XR-Nano
```

随后进行最小公共数据集验证：

```text
PIDray: YOLOv8n, YOLOv8s, XR-Nano
OPIXray: YOLOv8n, YOLOv8s, XR-Nano
```

在完成这些验证前，不要继续新增模型模块。
