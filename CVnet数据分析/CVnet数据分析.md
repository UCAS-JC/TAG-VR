# CVnet 数据分析

生成日期：2026-05-11  
数据路径：`/Volumes/AIRCAS_JC/data`  
分析口径：忽略 macOS `._*` 资源叉；将 `a2g` 与 `g2a` 视为两个检索协议；按文件名解析 `vehicle_id / camera / sequence / frame / instance`。

## 1. 结论摘要

- 磁盘中可解析 `.jpg` 图像文件共 **22,417** 个；按文件名去重后唯一图像为 **14,969** 个，对应既有材料中 CVPair/CVnet 的 **14,969** 张图像口径。
- 训练集在 `a2g` 与 `g2a` 两个协议目录中重复存放，重复 basename 数为 **7,448**，主要是 `bounding_box_train` 的协议副本。
- 训练集 **391** 个 ID，query/test 评测集 **503** 个 ID；训练与评测 ID 交集为 **0**，可以按现有目录直接构造 benchmark。
- `c0` 可视为地面/近景侧，`c1` 可视为空中/俯视侧：`a2g` 是空到地检索，`g2a` 是地到空检索。
- 数据清洗项：发现 **22,426** 个 `._*` 资源叉文件、**1** 个非图像残留文件；在非零填充 ID 候选文件中检测到 **88** 个扩展名为 `.jpg` 但文件头为 PNG 的图像。

## 2. 目录结构与协议

![CVnet 目录结构](assets/cvnet_analysis/directory_tree.png)

| 协议 | split | 图像数 | ID 数 | ID 范围 | camera 分布 |
| --- | --- | ---: | ---: | --- | --- |
| `a2g` | `query` | 503 | 503 | 0101-0911 | c1: 503 |
| `a2g` | `train` | 7,448 | 391 | 0001-0500 | c0: 1569, c1: 5879 |
| `a2g` | `gallery` | 1,513 | 503 | 0101-0911 | c0: 1513 |
| `g2a` | `query` | 503 | 503 | 0101-0911 | c0: 503 |
| `g2a` | `train` | 7,448 | 391 | 0001-0500 | c0: 1569, c1: 5879 |
| `g2a` | `gallery` | 5,002 | 503 | 0101-0911 | c1: 5002 |

协议解释：

- `a2g/query` 使用 `c1` 图像作为空中侧查询，`a2g/bounding_box_test` 使用 `c0` 图像作为地面侧 gallery。
- `g2a/query` 使用 `c0` 图像作为地面侧查询，`g2a/bounding_box_test` 使用 `c1` 图像作为空中侧 gallery。
- `bounding_box_train` 在两个协议目录中一致，包含同一批 391 个训练 ID 的 `c0+c1` 图像。

## 3. 数量分布

![split 图像数量](assets/cvnet_analysis/split_image_counts.png)

![ID 数量](assets/cvnet_analysis/id_counts.png)

![camera 分布](assets/cvnet_analysis/camera_distribution.png)

![每 ID 图像数量分布](assets/cvnet_analysis/images_per_id_histogram.png)

![frame 分布](assets/cvnet_analysis/frame_distribution.png)

frame 文件名统计显示，query 默认取 `00010`，gallery 从 `00020/00030/00040` 等帧开始扩展。训练集每个 ID 通常有更多帧，适合做 ID 级文本描述聚合；query/gallery 更适合做评测协议。

| 统计对象 | 高频 frame |
| --- | --- |
| train | 30: 782, 20: 782, 10: 782, 40: 776, 50: 396, 60: 348, 70: 339, 80: 332 |
| a2g gallery | 20: 503, 30: 503, 40: 497, 50: 6, 60: 3, 70: 1 |
| g2a gallery | 20: 503, 30: 503, 40: 501, 50: 494, 60: 409, 70: 401, 80: 388, 90: 360 |

## 4. 样例观察

![同 ID 跨视角样例](assets/cvnet_analysis/same_id_cross_view_examples.png)

上图展示同一车辆 ID 的地面近景与空中/高位视角。可见跨视角差异主要来自三个方面：尺度变化、可见部件变化、背景遮挡变化。文本标注应强调稳定属性，例如颜色、车型、车身轮廓、车窗/车顶/厢体等，而不应把单帧背景写成身份特征。

![质量问题样例](assets/cvnet_analysis/quality_issue_examples.png)

质量风险：

- **尺度差异大**：部分空中侧图像尺寸较小，车辆只占画面一部分；VLM 容易在品牌、细节、车灯等字段上幻觉。
- **背景与遮挡强**：停车场、车库、其他车辆、摩托车和柱体会干扰颜色/车型判断。
- **目标边界不稳定**：部分图片不是严格 tight crop，前景或邻车会进入画面，文本标注需要明确“主目标车辆”。
- **格式不统一**：存在 `.jpg` 扩展名但实际为 PNG 的图像，训练前应使用图像库读取并统一转码或在 dataloader 中容错。

## 5. 清洗建议

1. 复制一份干净索引，不直接修改原始外置盘数据。
2. 索引生成时排除 `._*`、`.baiduyun.uploading.cfg` 等非图像文件。
3. 用文件头而不是扩展名识别图像格式；必要时统一导出为 RGB JPEG 或 PNG。
4. 按 basename 去重统计全量规模，按协议路径保留 `a2g/g2a` 评测结构。
5. 对低分辨率、遮挡、邻车干扰样本加入 `uncertain` 和 `qa_status` 字段，避免自动标注过度确定。

## 6. Benchmark 使用建议

- 视觉 Re-ID baseline：沿用 `bounding_box_train` 训练，分别在 `a2g` 和 `g2a` 上报告 mAP、Rank-1、Rank-5、Rank-10。
- 文本 Re-ID baseline：按 vehicle ID 生成稳定文本描述，再构造 text-to-image / image-to-text 检索任务。
- 标注粒度：建议主数据采用 **ID 级描述 + 图像级结构化属性**。ID 级描述减少重复 caption，图像级属性保留视角、遮挡、可见部件差异。
- 质检策略：每个 ID 至少抽查 1 张空中侧与 1 张地面侧；低置信度样本进入人工复核队列。
