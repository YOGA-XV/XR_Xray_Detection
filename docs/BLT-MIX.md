**X射线违禁品检测数据增强方案文档**，包括物理动机、方法定义、公式、实现流程、参数、消融和实验设计。文档结构清晰、可复现，并突出创新点 **BLT-Mix**。

------

# SPXray / PIDray 数据增强方案文档（完整）

## 1. 设计目标

本数据增强方案面向 **X 射线违禁品检测任务**，设计目标：

1. 模拟 **行李内部真实重叠和遮挡**，改善小目标和细长目标检测精度；
2. 保留 **物理合理性**，符合 X 射线透射成像规律；
3. 提供 **可控难度调度**，便于消融和实验验证；
4. 与 **YOLOv8n-XR-Lite 模型体系**兼容，不增加推理开销。

核心创新：

> **BLT-Mix（Beer-Lambert Transmittance Mixup）**：在透射率近似域进行局部乘性合成，同时融合 hue_proxy 以保留高风险材料先验。

------

## 2. 增强方法分类

| 增强方法     | 核心思想                                    | 与 BLT-Mix 的关系  |
| ------------ | ------------------------------------------- | ------------------ |
| Linear MixUp | RGB 线性加权合成                            | 用作对比实验       |
| Copy-Paste   | 目标 patch 替换                             | 用作对比实验       |
| Mix-Paste    | 同类 patch 替换                             | 用作对比实验       |
| BGM          | 背景 patch mixup + pseudo-color             | 用作对比实验       |
| **BLT-Mix**  | 透射率域局部乘性叠加 + hue_proxy 高风险保留 | 创新点，论文主增强 |

------

## 3. BLT-Mix 方法定义

### 3.1 输入

- Base image: 
  $$
  (I_b \in [0,1]^{H\times W\times 3})
  $$
  
- Source patch/image: 
  $$
  (I_s \in [0,1]^{H\times W\times 3})
  $$
  
- Optional hue_proxy: 
  $$
  (H_b, H_s \in [0,1]^{1\times H\times W})
  $$
  
- Patch mask:
  $$
  (M \in {0,1}^{H\times W})
  $$
  
- Labels: 
  $$
  (Y_b, Y_s)
  $$
  

------

### 3.2 透射率近似

对伪色图像使用亮度通道或灰度归一化近似透射率：

[
$$
L=0.299R+0.587G+0.114B
$$
]

归一化：

[
$$
T = \text{clip}\Big(\frac{L-P_1(L)}{P_{99}(L)-P_1(L)+\epsilon},0,1\Big)
$$
]

> 对设备显示极性做校正，使大值对应弱衰减。

------

### 3.3 局部乘性合成公式

[
$$
T_{mix}=M\cdot(T_b\cdot T_s^\alpha)+(1-M)\cdot T_b
$$
]

- $$
  (\alpha \in [0.4,1.0])
  $$

   为有效厚度系数；

- 局部 mask (M) 可为 bbox mask、目标 mask 或遮挡区域。

等价光学密度形式：

[
$$
A=-\log(T+\epsilon), \quad A_{mix}=A_b+\alpha A_s, \quad T_{mix}=e^{-A_{mix}}
$$
]

------

### 3.4 hue_proxy 融合

[
$$
H_{mix}=M\cdot \max(H_b,H_s)+(1-M)\cdot H_b
$$
]

- 高风险材料保守保留；
- 避免线性平均削弱高 (Z_{eff}) 特征。

------

### 3.5 RGB 恢复策略

#### 方案 A（推荐，稳健）

- 只使用
  $$
  (T_{mix})
  $$
   调整亮度；

- 保留原始 
  $$
  RGB hue/saturation；
  $$

- 避免伪色过度干扰模型。

#### 方案 B（可选，消融）

- 将 source patch 的 
  $$
  RGB hue/saturation
  $$
   叠加到 mask 区域；

- 模拟真实贴入效果，但颜色材料可能不稳定；

- 用于消融实验。

------

## 4. Patch 选择与放置策略

### 4.1 Patch 权重（难度优先）

[
$$
w_j = \lambda_s w_{small} + \lambda_t w_{thin} + \lambda_o w_{occ} + \lambda_c w_{class}
$$
]

- 小目标权重：
  $$
  (w_{small}=1-\text{area}_j)
  $$
  
- 细长权重：
  $$
  (w_{thin}=\min(r_j/r_{max},1))
  $$
  
- 遮挡权重：
  $$
  (w_{occ} = \max_{k\neq j} IoU(b_j,b_k))
  $$
  
- 类别权重：
  $$
  (w_{class}=1/\sqrt{n_{c_j}})
  $$
  （SPXray 类别均衡可置 0）

- 推荐系数：
  $$
  (\lambda_s=0.3,\lambda_t=0.3,\lambda_o=0.4,\lambda_c=0)
  $$
  

------

### 4.2 Patch 放置模式

| 模式                  | 概率 | 描述                       |
| --------------------- | ---- | -------------------------- |
| Random Placement      | 0.3  | 随机位置                   |
| Near-object Placement | 0.4  | 与已有目标相邻             |
| Overlap Placement     | 0.3  | 与已有目标 IoU ∈ [0.1,0.7] |

------

## 5. 训练调度策略

### 5.1 阶段式概率

| 阶段 | Epoch % | BLT-Mix 概率 | (\alpha) 范围 |
| ---- | ------- | ------------ | ------------- |
| 初期 | 0–20%   | 0.1          | 0.4–0.6       |
| 中期 | 20–60%  | 0.3          | 0.5–0.8       |
| 后期 | 60–90%  | 0.4          | 0.6–1.0       |
| 收敛 | 90–100% | 0            | 关闭          |

------

### 5.2 与 Mosaic / CutMix 关系

- Mosaic 保留，但后期关闭；
- CutMix/Linear MixUp/Copy-Paste 作为消融实验；
- BLT-Mix 主增强，不影响推理开销。

------

## 6. BLT-Mix 流程（训练阶段）

```text
Input: base image I_b, source patch I_s, mask M, hue_proxy H_b/H_s
1. 基础几何增强（flip/scale/rotate/Mosaic）
2. BLT-Mix（透射率乘性叠加 + hue_proxy 融合）
3. PCN：伪色归一化
4. EGI：Scharr 边缘图生成
5. RGB + Edge 四通道输入模型
6. 计算检测损失
```

------

## 7. 参数建议（复现）

| 参数           | 建议值              |
| -------------- | ------------------- |
| BLT-Mix 概率   | 0.2–0.4             |
| Patch 数       | 1–3                 |
| α 范围         | 0.4–1.0             |
| Mask 类型      | bbox 或 object mask |
| Overlap IoU    | 0.1–0.7             |
| Hue Proxy 融合 | max                 |
| Color jitter   | 极弱                |

------

## 8. 实验验证设计

### 8.1 增强对比表

| Augmentation      | mAP50 | mAP50:95 | AP-small | AP-thin | Hidden AP | FPS  |
| ----------------- | ----- | -------- | -------- | ------- | --------- | ---- |
| Baseline          |       |          |          |         |           |      |
| Linear MixUp      |       |          |          |         |           |      |
| Copy-Paste        |       |          |          |         |           |      |
| Mix-Paste         |       |          |          |         |           |      |
| BGM               |       |          |          |         |           |      |
| **BLT-Mix, ours** |       |          |          |         |           |      |
| BLT-Mix + EGI     |       |          |          |         |           |      |

### 8.2 BLT-Mix 组件消融

| Method                    | Local Mask | α    | Hue Proxy | Overlap Sampling | mAP50:95 | Hidden AP | AP-thin |
| ------------------------- | ---------- | ---- | --------- | ---------------- | -------- | --------- | ------- |
| Linear MixUp              | ×          | ×    | ×         | ×                |          |           |         |
| Full-image Multiply       | ×          | ×    | ×         | ×                |          |           |         |
| Local BLT                 | ✓          | ×    | ×         | ×                |          |           |         |
| Local BLT + α             | ✓          | ✓    | ×         | ×                |          |           |         |
| Local BLT + α + hue_proxy | ✓          | ✓    | ✓         | ×                |          |           |         |
| Full BLT-Mix              | ✓          | ✓    | ✓         | ✓                |          |           |         |

### 8.3 α 参数敏感性

| α 范围    | mAP50:95 | Hidden AP | Visual Quality |
| --------- | -------- | --------- | -------------- |
| 0.2–0.5   |          |           | 过弱           |
| 0.4–0.7   |          |           | 自然           |
| 0.6–1.0   |          |           | 强遮挡         |
| scheduled |          |           | 推荐           |

------

## 9. 模块定位与论文创新点

- **创新点 1**：边缘引导伪色输入（PCN + EGI）
- **创新点 2**：轻量特征增强（EMA-Lite + AIFI-Lite）
- **创新点 3**：BLT-Mix 数据增强（透射率域乘性 + hue_proxy 高风险保留）
- **创新点 4**：轻量实时检测框架（YOLOv8n-XR-Lite）

------

## 10. 总结

1. BLT-Mix 是 X 射线专属物理增强，基于透射率域乘性叠加，结合 hue_proxy 高风险材料保留策略。
2. 可以替代 BGM-XR，作为论文主创新点之一。
3. 与 EGI/PCN 模块结合使用，可增强模型对隐藏、重叠和小目标检测能力。
4. 消融实验、参数敏感性实验和对比 BGM/Mix-Paste 都已设计。
5. 完整训练流程可在 YOLOv8n-XR-Lite 框架中复现，不影响推理效率。

------

这个文档可以直接放入论文方法章节、训练实现文档、实验设计计划，并作为创新点描述。