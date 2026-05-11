# 当前主线

## 冻结结论

当前候选主线为：

```text
XR-Nano = YOLOv8n + PCN + EGI
```

基于当前 SPXray 证据，XR-Nano 被冻结为候选主线。它在保持模型轻量的同时，引入了两个与 X 射线安检图像问题直接相关的输入侧改动：

1. PCN：伪彩色归一化。
2. EGI：Scharr 边缘引导的 4 通道输入。

## 为什么当前选择 XR-Nano

P0 汇总优先采用 `best.pt` 重新验证得到的结果。在最新 SPXray 验证结果中，A2-EGI 的 mAP50:95 略高于 XR-Nano，因此论文中不能写成“XR-Nano 是绝对 mAP 最优模型”。XR-Nano 被保留为候选主线，是因为它提供了更符合论文主张的综合平衡：

- mAP50:95 优于 XR-Lite、P2-Lite 和 XR-Plus。
- AP-thin 优于 YOLOv8n、A1-PCN、A2-EGI 以及后续结构增强版本。
- 端到端 FPS 高于 XR-Lite、P2-Lite 和 XR-Plus。
- 同时结合 PCN 与 EGI，使“伪彩色归一化 + 边缘结构引导”的方法主张更加完整。

因此，论文方向应表述为“面向伪彩色 X 射线安检图像的边缘结构引导输入增强方法”，而不是写成简单堆叠新模块。

## 其他模型的定位

YOLOv8n 和 YOLOv8s 保留为 baseline。

A1-PCN 和 A2-EGI 作为输入侧消融实验。

XR-Lite、P2-Lite 和 XR-Plus 保留为结构增强消融实验。基于当前 SPXray 证据，它们相对 XR-Nano 没有带来净收益，因此在 P0 阶段不提升为主模型。

## P0 冻结产物

P0 汇总包统一放在：

```text
runs/summary/
```

关键文件：

```text
runs/summary/model_registry.csv
runs/summary/spxray_main_results.csv
runs/summary/spxray_per_class_ap.csv
runs/summary/spxray_dataset_stats.csv
runs/summary/spxray_dataset_stats/
runs/summary/subset_metrics/
```

## 下一阶段边界

在完成下一轮验证闭环之前，不要继续新增模型模块。下一阶段应重点完成：

1. YOLOv8n 与 XR-Nano 的 3 seed 稳定性实验。
2. PIDray 最小公共数据集验证。
3. OPIXray 遮挡专项验证。
4. 在完成最小公共数据集验证后，再做鲁棒性实验。

BLT-MIX 仍作为后续数据增强消融实验，不纳入 P0 模型主张。
