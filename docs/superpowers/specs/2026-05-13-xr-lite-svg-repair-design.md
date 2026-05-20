# XR-Lite 结构图连线修订与 SVG 重绘规格

## 1. 任务目标

基于现有位图结构图：

- `G:\XR_Xray_Detection\papers\figures\XR-Lite.png`

修正其中与真实模型拓扑不一致或表达含糊的模块连接关系，并交付一份可编辑的矢量母版：

- `G:\XR_Xray_Detection\papers\figures\XR-Lite-corrected.svg`

## 2. 唯一拓扑依据

本次修订以当前工程中的模型定义为准：

- `G:\XR_Xray_Detection\ultralytics\cfg\models\v8\yolov8n-xr-lite.yaml`

相关结构说明文档仅作术语和表达辅助，不覆盖 YAML 的真实连接关系。

## 3. 版式与视觉策略

采用已确认的方案 A：

1. 保留原图的四段式论文图布局：
   - Input Preprocessing
   - YOLOv8n Backbone
   - PAN-FPN Neck
   - Detection Head
2. 保留原图的整体阅读方向、色彩分区和科研图语义。
3. 不做简单 PNG 矢量描摹，而是重新绘制为原生 SVG 元素，确保：
   - 文本可编辑；
   - 线条与框体可编辑；
   - 不依赖外链位图资源；
   - 后续可在 Illustrator、Inkscape、Figma 中继续修改。

## 4. 必须修正的连接关系

### 4.1 Top-down 主链

严格表达为：

1. `P5 / 32 (AIFI-enhanced)` -> `Upsample x2`
2. 与增强后的 `P4 / 16` 进行 `Concat`
3. 经 `C2f x3`
4. 再 `Upsample x2`
5. 与 `P3 / 8` 进行 `Concat`
6. 经 `C2f x3` 形成 `Neck P3 / 8`

### 4.2 P3 输出分流

`Neck P3 / 8` 必须同时：

1. 送入 `Detect P3 / 8`
2. 经 `Conv 3x3, s=2` 下采样，参与 bottom-up P4 融合

### 4.3 Bottom-up PAN 回流

严格表达为：

1. `Neck P3 / 8` -> `Conv 3x3, s=2`
2. 与 top-down 阶段的 P4 中间特征 `Concat`
3. 经 `C2fEMALite x3` 形成 `Neck P4 / 16`
4. `Neck P4 / 16` 同时：
   - 送入 `Detect P4 / 16`
   - 继续经 `Conv 3x3, s=2` 下采样到 P5 融合
5. 下采样后的 P4 路径与 `P5 / 32 (AIFI-enhanced)` 进行 `Concat`
6. 经 `C2fEMALite x3` 形成 `Neck P5 / 32`
7. 输出到 `Detect P5 / 32`

### 4.4 增强模块位置表达

保持并澄清：

1. Backbone P4 对应增强模块：
   - `C2fEMALite x6`
2. Backbone P5 上下文模块：
   - `SPPFAIFILite`
3. Neck P4 与 Neck P5 输出位置：
   - `C2fEMALite x3`

## 5. 允许的文字微调

允许为降低歧义而调整局部标签，例如：

- `P4 Top-down Intermediate`
- `Neck P3 / 8 (Top-down Output)`
- `Neck P4 / 16 (EMA-Lite Refined Output)`
- `Neck P5 / 32 (EMA-Lite Refined Output)`

但不改变模型命名，不新增与 YAML 不一致的结构解释。

## 6. 交付物

主交付：

- `papers/figures/XR-Lite-corrected.svg`

可选辅助交付：

- 如有必要，可追加一份同内容 PNG 预览，但它不是本次必需项。

## 7. 验收标准

最终 SVG 需满足：

1. 结构连接与 `yolov8n-xr-lite.yaml` 一致；
2. PAN-FPN 的 top-down 与 bottom-up 流向清晰；
3. 三个检测头的来源关系表达正确；
4. 文本、箭头、模块框体均为 SVG 可编辑元素；
5. 视觉风格延续原图，不改成完全不同的版式；
6. 文件可正常打开，无缺失资源。

## 8. 非目标

以下内容不在本次范围内：

1. 不重画 `XR-Plus`；
2. 不改论文方法定义；
3. 不调整实验结论；
4. 不额外生成 draw.io、PPTX 或 Figma 文件；
5. 不把图改造成仓库中现有脚本的另一套卡片式风格。
