# SPXray 数据统计分析

## 1. 文档目的

本文档对应 `docs/2项目实施步骤.md` 中的阶段 3：做 SPXray 数据统计表。

阶段 3 的目标是补齐论文级数据统计，使后续实验结果不只是报告模型指标，还能说明数据集本身的目标尺度、形态、重叠程度和类别分布特征。

本文档使用的统计产物来自：

```text
runs/summary/spxray_dataset_stats.csv
runs/summary/spxray_dataset_stats/summary.json
runs/summary/spxray_dataset_stats/split_summary.csv
runs/summary/spxray_dataset_stats/class_distribution.csv
runs/summary/spxray_dataset_stats/bbox_area_histogram.csv
runs/summary/spxray_dataset_stats/aspect_ratio_histogram.csv
```

## 2. 统计定义

### 2.1 目标尺度定义

目标尺度按 bbox 面积占整图面积的比例定义：

```text
area = bbox_width * bbox_height / (image_width * image_height)
```

| 类型 | 定义 |
| --- | --- |
| small | area < 0.01 |
| medium | 0.01 <= area < 0.05 |
| large | area >= 0.05 |

### 2.2 细长目标定义

细长目标按 bbox 长宽比定义：

```text
max(width, height) / min(width, height) > 3
```

### 2.3 重叠目标定义

SPXray 当前没有官方 hidden 标注，因此本文档不使用 Hidden AP / Hidden Recall 这类命名。SPXray 上只统计 overlap proxy：

```text
overlap = max_j Area(b_i intersection b_j) / Area(b_i)
```

本文档报告：

1. overlap@0.3
2. overlap@0.5

## 3. 数据集整体规模

SPXray 当前项目内统计结果：

| Split | Images | Instances | Missing labels |
| --- | ---: | ---: | ---: |
| train | 10982 | 14413 | 0 |
| val | 1373 | 1794 | 0 |
| test | 1373 | 1824 | 0 |
| total | 13728 | 18031 | 0 |

结论：

1. 训练集、验证集、测试集都有完整标签文件。
2. 总实例数为 18031。
3. 测试集与验证集图像数量一致，便于后续固定评估协议。

## 4. 目标尺度、细长目标与重叠统计

### 4.1 分 split 统计

| Split | Images | Instances | Small | Small(%) | Medium | Medium(%) | Large | Large(%) | Thin | Thin(%) | Overlap@0.3 | Overlap@0.3(%) | Overlap@0.5 | Overlap@0.5(%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 10982 | 14413 | 930 | 6.45 | 4312 | 29.92 | 9171 | 63.63 | 2363 | 16.39 | 2898 | 20.11 | 1872 | 12.99 |
| val | 1373 | 1794 | 107 | 5.96 | 558 | 31.10 | 1129 | 62.93 | 283 | 15.77 | 372 | 20.74 | 232 | 12.93 |
| test | 1373 | 1824 | 118 | 6.47 | 553 | 30.32 | 1153 | 63.21 | 277 | 15.19 | 403 | 22.09 | 261 | 14.31 |

### 4.2 全数据集统计

| 指标 | 数量 | 占实例比例 |
| --- | ---: | ---: |
| small | 1155 | 6.41% |
| medium | 5423 | 30.08% |
| large | 11453 | 63.52% |
| thin | 2923 | 16.21% |
| overlap@0.3 | 3673 | 20.37% |
| overlap@0.5 | 2365 | 13.12% |

结论：

1. SPXray 以 large object 为主，large 占 63.52%。
2. small object 占 6.41%，比例不高，但仍是需要单独报告的难样本指标。
3. thin object 占 16.21%，足以支撑 AP-thin / Recall-thin 作为论文指标。
4. overlap@0.3 占 20.37%，overlap@0.5 占 13.12%，说明 SPXray 可以作为 overlap proxy 分析数据集，但不能称为 hidden 数据集。

## 5. 每类别实例数与图像数

| ID | Class | Train inst. | Train imgs | Val inst. | Val imgs | Test inst. | Test imgs | Total inst. | Total imgs |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | Baton | 1200 | 1131 | 148 | 143 | 157 | 145 | 1505 | 1419 |
| 1 | Plier | 1198 | 1039 | 168 | 142 | 133 | 116 | 1499 | 1297 |
| 2 | Hammer | 1222 | 1192 | 142 | 135 | 144 | 142 | 1508 | 1469 |
| 3 | Powerbank | 1200 | 1148 | 154 | 147 | 156 | 150 | 1510 | 1445 |
| 4 | Scissors | 1191 | 1121 | 166 | 157 | 154 | 140 | 1511 | 1418 |
| 5 | Wrench | 1215 | 1084 | 135 | 119 | 145 | 131 | 1495 | 1334 |
| 6 | Gun | 1201 | 947 | 143 | 124 | 175 | 132 | 1519 | 1203 |
| 7 | Bullet | 1182 | 1176 | 162 | 162 | 155 | 155 | 1499 | 1493 |
| 8 | Sprayer | 1179 | 1167 | 146 | 144 | 170 | 167 | 1495 | 1478 |
| 9 | HandCuffs | 1209 | 1206 | 156 | 156 | 136 | 136 | 1501 | 1498 |
| 10 | Knife | 1224 | 1016 | 132 | 112 | 139 | 116 | 1495 | 1244 |
| 11 | Lighter | 1192 | 1181 | 142 | 140 | 160 | 158 | 1494 | 1479 |

## 6. 类别长尾程度

按总实例数统计：

| 指标 | 数值 |
| --- | ---: |
| 最大类别实例数 | 1519 |
| 最小类别实例数 | 1494 |
| 最大/最小比例 | 1.017 |

结论：

1. SPXray 的类别实例分布非常均衡。
2. 当前 SPXray 上的消融实验更适合用来判断模型结构本身是否有效。
3. 长尾类别问题不应作为 SPXray 阶段的主要论文叙事，应留到 PIDray / SIXray 等公共数据集上讨论。

## 7. bbox 面积分布

| Area bin | Count | Percent |
| --- | ---: | ---: |
| [0.0,0.001) | 12 | 0.07% |
| [0.001,0.0025) | 121 | 0.67% |
| [0.0025,0.005) | 309 | 1.71% |
| [0.005,0.01) | 713 | 3.95% |
| [0.01,0.02) | 1591 | 8.82% |
| [0.02,0.05) | 3832 | 21.25% |
| [0.05,0.1) | 5457 | 30.26% |
| >=0.1 | 5996 | 33.25% |

结论：

1. bbox 面积分布进一步说明 SPXray 不是一个极端小目标数据集。
2. 小目标指标仍然需要报告，但不能把 SPXray 描述成主要由小目标构成的数据集。

## 8. 长宽比分布

| Aspect-ratio bin | Count | Percent |
| --- | ---: | ---: |
| [1.0,1.5) | 7238 | 40.14% |
| [1.5,2.0) | 3912 | 21.70% |
| [2.0,3.0) | 3956 | 21.94% |
| [3.0,5.0) | 2182 | 12.10% |
| [5.0,10.0) | 710 | 3.94% |
| >=10.0 | 33 | 0.18% |

结论：

1. 长宽比大于 3 的 thin object 合计 2923 个，占 16.21%。
2. AP-thin 与 Recall-thin 是合理的专项指标。
3. Knife、Scissors、Wrench 等类别在论文中适合作为细长目标可视化案例。

## 9. 阶段 3 结论

阶段 3 已经完成。

SPXray 的论文级数据统计结论如下：

1. SPXray 总体规模为 13728 张图像、18031 个实例。
2. 类别实例分布非常均衡，最大/最小类别实例数比例仅为 1.017。
3. small object 占 6.41%，适合报告 AP-small，但不是 SPXray 的主导难点。
4. thin object 占 16.21%，可以支撑 AP-thin / Recall-thin。
5. overlap@0.3 占 20.37%，overlap@0.5 占 13.12%，可以作为 overlap proxy 指标。
6. SPXray 没有官方 hidden 标注，因此在 SPXray 上不能写 Hidden AP，只能写 overlap AP。

## 10. 后续使用方式

论文或开题报告中建议这样使用本阶段结果：

1. 在 Dataset 小节放数据集规模表和类别分布表。
2. 在 Evaluation Metrics 小节解释 small / thin / overlap 的定义。
3. 在 SPXray Results 小节报告 AP-small、AP-thin、AP-overlap@0.3、AP-overlap@0.5。
4. 在 Limitations 中说明 SPXray 的 hidden 标注缺失，因此 hidden 场景需要 PIDray / OPIXray 继续验证。
