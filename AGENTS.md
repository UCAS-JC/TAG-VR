# TAG-VR 项目 Agent 说明

本文件用于帮助后续 agent 快速理解本项目背景、资料位置、数据口径、标注规范、baseline 方向和工作边界。

## 项目概览

- 项目名称：`TAG-VR`
- 英文全称：`Text-enhanced Aerial-Ground Vehicle Retrieval`
- 投稿目标：整理为 AAAI 2027 论文投稿材料。
- 核心目标：基于 CVnet/CVPair 真实无人机-地面车辆跨视角数据，补充文本标注，并引入经过处理的空地跟踪、目标识别和协同感知数据集作为扩展，形成文本增强的空地车辆检索 benchmark。
- 任务定位：`TAG-VR` 包含 Re-ID，但不局限于纯 image-to-image Re-ID。它应支持 visual retrieval、text-to-image retrieval、image-to-text retrieval、text-guided cross-view retrieval、attribute/type retrieval 和跨数据源泛化分析。
- 主要贡献口径：benchmark 定义、文本标注协议、多源空地车辆配对扩展、跨视角图文对齐 baseline 与消融实验，而不是简单给车辆图像生成 caption。

## 资料地图

- `CVnet数据分析.md`：CVnet/CVPair 真实数据统计、协议解释、质量问题和 benchmark 使用建议。
- `动机.md`：论文动机。
- `TAG-VR数据集制作完整工作流程说明.md`：数据集制作流程说明。
- `assets/cvnet_analysis/`：CVnet 数据分析图，包括 split 统计、ID 统计、camera 分布、质量问题样例和同 ID 跨视角样例。

## 真实数据基础

- 真实数据路径：`/Volumes/AIRCAS_JC/data`。
- 数据来源口径：CVnet/CVPair 真实空地车辆跨视角数据。
- 唯一图像规模：约 14,969 张。
- 车辆 ID：894 个。
- 训练集：391 个 ID。
- 评测集：503 个 ID。
- 训练和评测 ID 无交集。
- `c0` 可视为 ground / near-view 侧。
- `c1` 可视为 UAV / high-view 侧。
- `a2g` 表示 aerial-to-ground retrieval：UAV image query 到 ground gallery。
- `g2a` 表示 ground-to-aerial retrieval：ground image query 到 UAV gallery。
- `bounding_box_train` 在 `a2g` 和 `g2a` 两个协议目录中重复存放，统计时需要按 basename 去重，训练协议上仍保留原结构。

已知质量问题：

- 存在 macOS `._*` 资源叉和百度云残留文件，生成索引时必须过滤。
- 部分 `.jpg` 扩展名文件实际为 PNG，读取时应根据文件头或图像库识别真实格式。
- UAV 侧目标较小，低分辨率、小目标和遮挡样本应允许 `uncertain`。
- 地面侧可能包含邻车、柱体、摩托车、车库背景等干扰。
- 同一 ID 可能与停车位、道路纹理、车库柱体或邻车布局绑定，需关注背景捷径。

## 多源扩展计划

扩展数据来自已发布或可获取的空地跟踪、目标识别、协同检测、协同感知或空地双视角跟踪数据集。它们不能直接写成 `TAG-VR` Re-ID 数据，必须经过车辆筛选、空地视角图像抽取、身份关联核验、文本标注和 QA 后，才能作为 `TAG-VR` 的一部分。

候选数据源可以包括 V2U4Real、AGC-Drive、Griffin、AGVOT 等空地双视角或空地协同数据集，具体是否纳入以数据可获取性、许可、图像质量、车辆类别比例、跨视角关联可靠性和标注 schema 为准。不要把某一个数据源写成唯一扩展来源。

扩展数据准入条件：

- 同一数据源内必须包含 UAV / aerial 视角和 ground / vehicle / road-side / ground-camera 视角。
- 必须能筛选出车辆目标，例如 `car`、`truck`、`bus`、`van`、`vehicle`。
- 必须保留原始 `source_dataset`、版本、下载时间、许可状态、原始任务、传感器类型和 annotation schema。
- 必须保留或重建 `vehicle_id`，并记录关联依据，例如 global ID、track ID、timestamp、pose、3D 位置、bbox overlap 或人工复核。
- 只有 `global_id_verified` 或可靠 `tracklet_verified` 的样本可进入主检索协议；`weak_association` 只能作为弱监督或附录分析，`unverified` 不进入主 benchmark。
- 扩展 ID 不得与 CVnet/CVPair ID 混用，应使用数据源前缀或独立 namespace。

标准转换流程：

1. 数据源核查：记录下载入口、版本、许可、公开状态、原始任务、传感器类型和标注字段。
2. 类别筛选：只保留车辆相关目标，剔除行人、动物、普通物体和类别不确定样本。
3. 视角拆分：统一映射为 `uav`、`ground_vehicle`、`road_side`、`ground_camera` 等 `view_source`。
4. crop 导出：基于 2D bbox 或 3D bbox 投影导出车辆 crop，同时保留 full frame、bbox、timestamp、pose、scene_id、track_id 和原始标注索引。
5. 空地配对：使用原始 global ID 优先；没有 global ID 时，结合时间同步、空间位置、轨迹连续性和人工复核建立 UAV-ground 车辆配对。
6. 可靠性分级：每条关联标记为 `global_id_verified`、`tracklet_verified`、`weak_association` 或 `unverified`。
7. 文本标注：采用 ID 级稳定描述和图像级可见属性，不把道路纹理、停车位、柱体或邻车布局写成车辆身份特征。
8. 数据划分：按 source、scene、vehicle_id 和 tracklet 防止泄漏；训练、验证、测试划分必须记录随机种子和规则。

推荐目录结构：

```text
tag_vr_multisource/
  cvnet_index/
    cvnet_clean_index.jsonl
    cvnet_quality_notes.csv
  source_audit/
    source_audit.md
    source_schema_notes.md
  images/
  full_frames/
  metadata/
    metadata.jsonl
    identity_links.jsonl
    split.json
  annotations/
    annotations_id.jsonl
    annotations_image.jsonl
  qa/
    qa_report.md
    low_quality_samples.csv
```

扩展图像命名建议：

```text
{source_dataset}_{vehicle_id}_{view_source}_{sequence}_{frame:06d}_{instance:02d}.jpg
```

示例：

```text
v2u4real_V000123_uav_s01_000120_00.jpg
v2u4real_V000123_ground_vehicle_s01_000120_00.jpg
```

## 可用API信息
测试用：TURINGAI_BASE_URL="https://turingai.plus/v1"
TURINGAI_API_KEY="sk-8ctQRo2M2EtLVIBkRq2uvhMMDD5tYhfx3wf1UDJfo2gH1oO8"

真实大规模标注实验使用的API目前待定。

## 文本标注规范

文本标注分两层：

- ID 级稳定描述：每个车辆 ID 一条，强调跨视角稳定属性。
- 图像级属性描述：每张图一条，记录当前视角下可见的方向、部件、遮挡、背景和不确定项。

必须保留：

- `description_zh`
- `description_en`
- `vehicle_id`
- `view_source`
- `camera_id`
- `color`
- `vehicle_type`
- `orientation`
- `visible_parts`
- `occlusion`
- `scene_context`
- `confidence`
- `qa_status`

推荐属性：

- 颜色、粗粒度车型、车身轮廓、车顶结构、车窗、厢体、行李架、特殊贴纸、可见部件、遮挡程度。

禁止或谨慎项：

- 不自动生成品牌。
- 不自动生成具体型号。
- 不记录真实车牌号作为身份特征。
- 不描述图像中不可见的细节。
- 不把背景、停车位、道路纹理、邻车或车库柱体写成车辆身份特征。

不确定性规则：

- 小目标、低光、雨雾、遮挡、颜色难辨、目标边界不稳定时使用 `uncertain`。
- 低置信度样本进入 `manual_review`。
- 明显错误但可修正样本标记为 `fixed`。
- 主目标不可辨认、ID 不一致或 crop 中主目标无法区分时标记为 `drop`。

VLM 辅助标注要求：

- 可使用 Qwen3-VL 等 VLM 做初始标注，但必须保留 prompt、模型版本、API usage、人工修改记录和 QA 状态。
- 真实数据以图像可见内容为准。
- 扩展数据若提供 bbox、track ID、类别、视角、位姿或时间戳等元数据，应优先保留为结构化字段；文本描述仍以图像可见内容为准。
- 每个 ID 建议输入多张代表图生成稳定 ID 描述，避免逐图 caption 风格漂移。

## Benchmark 与 Baseline

`TAG-VR` 应保留传统 Re-ID 指标，同时扩展到文本增强检索任务。

| 任务 | 查询 | Gallery | 指标 |
| --- | --- | --- | --- |
| Visual a2g | UAV image | Ground images | mAP, Rank-1/5/10 |
| Visual g2a | Ground image | UAV images | mAP, Rank-1/5/10 |
| Text-to-ground | ID/text description | Ground images | mAP, Recall@K |
| Text-to-UAV | ID/text description | UAV images | mAP, Recall@K |
| Image-to-text | Image | ID/text descriptions | Recall@K |
| Text-guided cross-view retrieval | Image + text | Cross-view images | mAP, Rank-K |
| Attribute/type retrieval | Color/type/orientation attributes | Images | Recall@K, attribute accuracy |
| Cross-source generalization | One source image/text | Held-out source images | mAP, Rank-K, Recall@K |

最低 baseline 组合：

- 视觉 Re-ID baseline：ResNet50、ViT 或 TransReID 类 backbone，在 `a2g` 和 `g2a` 上报告 mAP、Rank-1、Rank-5、Rank-10。
- 图文 baseline：CLIP-style dual encoder，用 ID 描述与图像做对比学习，报告 text-to-image Recall@K 和 cross-view mAP。
- 文本引导 baseline：融合图像特征与文本特征后做跨视角检索。
- 跨视角图文对齐 baseline：暂用 `[Baseline Name]`、`[cross-view vision-language alignment module]`、`[attribute-guided alignment objective]` 或 `[view-aware contrastive learning strategy]` 作为占位，后续明确技术路线后再替换。
- 属性/车型检索 baseline：基于 `vehicle_type`、颜色和车身结构属性报告 type-level Recall@K 与 attribute accuracy。
- 多源扩展消融：CVnet/CVPair-only、extension-only、CVnet/CVPair+extensions，并区分是否使用文本监督、是否使用属性监督、是否使用跨视角对齐模块。

## 论文口径与风险

推荐贡献表述：

- 提出 `TAG-VR`，一个文本增强的无人机-地面车辆检索 benchmark。
- 在 CVnet/CVPair 真实空地同 ID 数据上补充 ID 级自然语言描述、图像级属性和 QA 元数据。
- 引入经过处理的空地跟踪、目标识别和协同感知数据集作为扩展，通过车辆筛选、跨视角配对、身份关联核验和统一文本标注，形成多源空地车辆检索样本。
- 设计 VLM 辅助标注、结构化 schema、人工复核和不确定性建模流程。
- 提供面向空地跨视角图文对齐的 baseline，并报告传统 Re-ID、文本检索、文本引导跨视角检索、属性检索和跨数据源泛化实验。

谨慎表述：

- 可以说 `one of the first text-enhanced benchmarks for aerial-ground vehicle retrieval`，前提是投稿前完成系统相关工作检索。
- 不应直接声称“第一个”，除非正式检索确认没有同类工作。
- 不应把空地跟踪、目标识别或协同感知原始数据直接称为 Re-ID 数据；必须强调它们是经过车辆筛选、空地配对、身份核验和文本标注后的扩展数据。
- 对 V2U4Real、AGC-Drive、Griffin、AGVOT 等具体数据源的表述，必须以实际下载内容、许可和 schema 核验为准。

主要风险：

- 多源域差：不同数据源的采集平台、相机参数、场景、标注粒度和身份关联可靠性可能不一致。
- 身份关联误差：跟踪或目标识别数据集未必天然提供跨平台 Re-ID 身份，需防止错误合并车辆 ID。
- VLM 幻觉：小目标和遮挡情况下容易误判颜色、车型或车顶结构。
- 背景泄漏：模型可能利用停车位、道路纹理、车库柱体和邻车布局完成检索。
- 文本粒度不一致：不同标注者或 VLM prompt 可能产生不同描述风格。
- 数据许可：CVnet/CVPair 原始数据和所有扩展数据源均需要确认发布许可、引用要求和再分发限制。

## 后续 Agent 工作守则

- 不要直接移动、删除或改写 `/Volumes/AIRCAS_JC/data` 原始数据。
- 对真实数据优先生成干净索引，例如 `cvnet_clean_index.jsonl`，不要修改原始目录结构。
- 统计规模时区分路径数和唯一 basename 数；训练目录在 `a2g` 与 `g2a` 中有重复副本。
- 扩展数据必须先完成 source audit、schema 解析、车辆筛选、空地配对和身份可靠性分级，再进入统一标注。
- 不要跨场景强行合并扩展数据中的车辆身份；无法证明同一车辆的样本只能作为弱关联样本或剔除。
- 任何生成脚本都应记录输入路径、输出路径、依赖版本、随机种子和配置文件。
- 任何 VLM 标注流程都应保留 prompt、模型名、模型版本、调用时间、usage、错误重试和人工修订记录。
- baseline 实验必须分别报告 `a2g` 和 `g2a`，不要只给合并结果。
- 文档中公开名称和 benchmark 总称使用 `TAG-VR` / `Text-enhanced Aerial-Ground Vehicle Retrieval`；技术任务说明中可以保留 Re-ID。
- 面向论文写作时，强调 `TAG-VR` 是文本增强空地车辆检索 benchmark，而不是普通车辆 caption 数据集。
