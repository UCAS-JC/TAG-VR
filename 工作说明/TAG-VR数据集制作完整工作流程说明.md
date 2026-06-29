# TAG-VR 数据集制作完整工作流程说明

生成日期：2026-06-24  
适用项目：`TAG-VR` / `Text-enhanced Aerial-Ground Vehicle Retrieval`  
适用对象：负责数据整理、数据转换、文本标注、质量检查和交付打包的同学或后续 agent

## 1. 工作定位

`TAG-VR` 的目标是构建一个文本增强的无人机-地面车辆检索 benchmark。数据集以 CVPair/CVnet 真实空地车辆 Re-ID 数据为核心，补充 ID 级自然语言描述、图像级结构化属性和 QA 元数据；同时引入经过处理的空地跟踪、目标识别、协同检测、协同感知或空地双视角跟踪数据集作为扩展。

外部扩展数据源不能直接写成 Re-ID 数据，必须经过车辆筛选、空地视角图像抽取、身份关联核验、文本标注和质量检查后，才能作为 `TAG-VR` 的一部分。

数据来源分为两层：

| 层级 | 数据源 | 定位 | 是否直接作为 TAG-VR 数据 |
| --- | --- | --- | --- |
| 核心真实 Re-ID | CVPair / CVnet | 主 benchmark，已有空地同 ID 协议 | 是，但需清洗索引、补文本和 QA |
| 多源空地扩展 | V2U4Real、AGC-Drive、Griffin、AGVOT 等候选空地数据源 | 由空地跟踪、目标识别、协同感知数据转换而来 | 否，需转换、核验和标注 |

关键原则：

- 不直接移动、删除或改写原始数据。
- 扩展数据必须记录来源、版本、许可、原始任务和 schema。
- 只有能证明同一车辆的样本才合并为同一 `vehicle_id`。
- 不跨场景强行合并身份；证据不足的样本只能作为弱关联、待复核或剔除。
- 文本标注服务于跨视角稳定语义，不是逐图自由 caption。
- 背景可以记录为 `scene_context`，但不能写成车辆身份特征。

## 2. 工作包与产出

| 工作包 | 主要任务 | 关键产出 | 验收要点 |
| --- | --- | --- | --- |
| WP0 需求与 schema 对齐 | 确认数据源范围、字段、命名、split 和 QA 状态枚举 | `configs/schema_mapping.yaml`、字段说明 | 后续脚本和标注都按同一 schema |
| WP1 CVPair 真实数据整理 | 读取原始目录，过滤无效文件，解析 ID、视角和协议，生成干净索引 | `cvpair_index/cvpair_clean_index.jsonl`、`cvpair_quality_notes.csv` | 不修改原始数据，路径可追溯 |
| WP2 扩展数据源核查 | 核查候选数据集下载状态、许可、传感器、图像、bbox、track/global ID | `source_audit/*.md`、`source_schema_notes.md` | 无图像或许可不明时不生成伪数据 |
| WP3 车辆筛选与 crop 导出 | 筛选车辆类，导出 UAV 和地面视角 crop，保留 full frame 引用 | `images/`、`full_frames/`、`metadata/metadata.jsonl` | 每个 crop 可追溯到原始帧和 bbox |
| WP4 空地配对与身份核验 | 沿用 global ID，或用时间、空间、轨迹和人工复核建立车辆关联 | `metadata/identity_links.jsonl`、身份统计 | 明确 `global_id_verified`、`tracklet_verified`、`weak_association`、`unverified` |
| WP5 文本标注与复核 | 生成 ID 级稳定描述和图像级属性描述，复核颜色、车型、视角和遮挡 | `annotations/annotations_id.jsonl`、`annotations/annotations_image.jsonl` | 不写不可见属性，不写品牌型号和真实车牌 |
| WP6 Benchmark split 与任务索引 | 构造 visual、text、attribute、cross-source 等任务所需 query/gallery 文件 | `splits/*.json`、`benchmarks/*.jsonl` | 训练、验证、测试无 ID 泄漏 |
| WP7 QA 与交付打包 | 脚本检查、人工抽查、质量报告和最终目录归档 | `qa/qa_report.md`、`qa/low_quality_samples.csv`、README | JSONL 可解析，字段完整，低质样本有处理状态 |

## 3. CVPair 真实数据整理要求

CVPair/CVnet 是 `TAG-VR` 的核心真实 Re-ID 数据。处理目标是生成干净索引和文本标注入口，不改写原始目录。

### 3.1 已知数据口径

- 唯一图像规模：约 14,969 张
- 车辆 ID：894 个
- 训练集：391 个 ID
- 评测集：503 个 ID
- 训练和评测 ID 无交集
- `c0` 可视为 ground / near-view 侧
- `c1` 可视为 UAV / high-view 侧
- `a2g` 表示 aerial-to-ground retrieval：UAV image query 到 ground gallery
- `g2a` 表示 ground-to-aerial retrieval：ground image query 到 UAV gallery
- `bounding_box_train` 在 `a2g` 和 `g2a` 两个协议目录中重复存放，统计唯一图像时按 basename 去重，训练协议上仍保留原结构

### 3.2 清洗索引要求

必须完成：

1. 遍历 `a2g` 和 `g2a` 两套协议目录。
2. 确认 `query`、`bounding_box_train`、`bounding_box_test` 对应的 split 和视角来源。
3. 过滤 macOS `._*` 资源叉、百度云残留文件、隐藏文件和非图像文件。
4. 使用图像库或文件头识别真实格式，记录 `.jpg` 扩展名但实际为 PNG 的样本。
5. 按文件名解析 `vehicle_id`、`camera_id`、`sequence`、`frame`、`instance`。
6. 保留原始相对路径和协议路径，避免只保存复制后的路径。
7. 每个 ID 至少抽查 1 张地面侧和 1 张空中侧图像，检查车辆身份、遮挡、颜色和车型是否适合文本标注。

已知质量问题必须进入 `cvpair_quality_notes.csv` 或 QA 报告：

- UAV 侧目标较小，低分辨率、小目标和遮挡样本应允许 `uncertain`。
- 地面侧可能包含邻车、柱体、摩托车、车库背景等干扰。
- 同一 ID 可能与停车位、道路纹理、车库柱体或邻车布局绑定，需关注背景捷径。
- 牌照不可见或不可用时，不应把车牌作为身份描述依据。

### 3.3 CVPair index 字段

`cvpair_index/cvpair_clean_index.jsonl` 每行对应一张可用图像，建议字段如下：

```json
{
  "dataset": "tag_vr",
  "source_dataset": "cvpair",
  "source_domain": "real",
  "protocol": "a2g",
  "split": "query",
  "vehicle_id": "cvpair_0101",
  "raw_vehicle_id": "0101",
  "image_path": "cvpair/a2g/query/0101_c1s2_00010_00.jpg",
  "original_path": "/Volumes/AIRCAS_JC/data/a2g/query/0101_c1s2_00010_00.jpg",
  "camera_id": "c1",
  "view_source": "uav",
  "sequence": "s2",
  "frame": 10,
  "instance": 0,
  "file_format": "png",
  "file_format_checked": true,
  "duplicate_basename": false,
  "qa_status": "raw"
}
```

## 4. 空地跟踪识别数据集处理要求

本节适用于 V2U4Real、AGC-Drive、Griffin、AGVOT 以及后续可能引入的空地跟踪、目标识别、协同检测、协同感知或空地双视角跟踪数据集。

### 4.1 数据源准入条件

候选数据源进入处理前必须满足或完成核查：

- 同一数据源内包含 UAV / aerial 视角和 ground / vehicle / road-side / ground-camera 视角。
- 原始数据中存在图像或可导出的图像帧，不能只有不可视化的点云或统计指标。
- 原始标注至少提供车辆类别和 bbox；如果没有 bbox，必须有可靠方式生成 crop。
- 车辆目标类别可筛选，例如 `car`、`truck`、`bus`、`van`、`vehicle`。
- 具有可用于关联的线索，例如 global ID、track ID、timestamp、pose、3D 位置、同步帧、平台标定或人工可复核画面。
- 许可、引用和再分发限制明确。

不满足准入条件时，只输出 source audit，不生成 `TAG-VR` 主数据。

### 4.2 数据源核查记录

每个数据源至少生成一个 audit 文件，例如：

```text
source_audit/
  v2u4real_audit.md
  agc_drive_audit.md
  griffin_audit.md
  agvot_audit.md
```

audit 必须记录：

- 数据集名称、版本、下载地址、下载日期、论文或项目链接。
- 许可、引用要求、是否允许再分发图像或只允许发布标注。
- 原始任务类型：tracking、detection、recognition、cooperative perception、VOT 等。
- 可用传感器和视角：UAV、ground vehicle、road-side、ground camera 等。
- 图像格式、帧率、分辨率、split、目录结构。
- 标注字段：类别、2D bbox、3D bbox、timestamp、pose、track ID、global ID。
- 是否可直接建立跨视角车辆关联。
- 已知限制、无法下载内容和暂缓处理原因。

### 4.3 车辆筛选要求

只保留车辆相关类别：

- `car`
- `truck`
- `bus`
- `van`
- `vehicle`
- `pickup`
- `light_truck`
- `box_truck`
- 数据源中等价的车辆类标签

必须剔除：

- 行人、骑行者、动物、普通物体、交通标志等非车辆目标。
- 类别无法确认且无法人工复核的目标。
- bbox 严重错误、主目标不可辨认或 crop 中多车混淆的目标。

### 4.4 空地配对与身份可靠性

扩展数据必须输出 `metadata/identity_links.jsonl`。身份可靠性分为四级：

| 可靠性 | 判定条件 | 可用于 |
| --- | --- | --- |
| `global_id_verified` | 原始数据提供跨平台或跨帧 global ID，或经 3D 同步和人工复核强确认 | 实例级 Re-ID、text-to-image、text-guided cross-view |
| `tracklet_verified` | 同一序列内 track ID 连续可靠，但不能证明跨序列同一身份 | tracklet retrieval、同序列分析、训练增强 |
| `weak_association` | 时间、空间或外观证据支持，但未强确认 | 辅助训练、错误分析、人工复核队列 |
| `unverified` | 无法确认同一车辆，或只有孤立检测框 | 不进入主 benchmark |

禁止把 `weak_association` 或 `unverified` 样本混入主测试集。论文和报告必须区分实例级 ID、tracklet 级 ID 和弱关联样本。

## 5. 转换为 TAG-VR 的标准流程

扩展数据源统一按以下流程转换。

1. 数据源核查：记录 `source_dataset`、版本、下载时间、许可、公开状态、原始任务、传感器类型和 annotation schema。
2. 字段映射：建立原始字段到 TAG-VR 字段的映射，写入 `configs/schema_mapping.yaml`。
3. 类别筛选：只保留车辆相关目标，剔除非车辆目标和不可确认类别。
4. 视角拆分：将传感器映射为 `uav`、`ground_vehicle`、`road_side`、`ground_camera` 等 `view_source`。
5. crop 导出：基于 2D bbox 或 3D bbox 投影导出车辆 crop，同时保留 full frame path、bbox、timestamp、pose、scene_id、track_id 和原始标注索引。
6. 空地配对：优先沿用原始 global ID；没有 global ID 时，结合时间同步、3D 空间位置、平台位姿、track ID、运动轨迹和人工复核生成 `vehicle_id`。
7. 可靠性分级：每个身份关联标记为 `global_id_verified`、`tracklet_verified`、`weak_association` 或 `unverified`。
8. 文本标注：生成 ID 级稳定描述和图像级可见属性。
9. split 构造：按 source、scene、vehicle_id 和 tracklet 防止泄漏；记录随机种子和划分规则。
10. QA 与打包：输出可解析 JSONL、QA report、低质样本清单、转换脚本版本和人工修订记录。

关键执行规则：

- 不跨源复用 `vehicle_id`。扩展 ID 必须带数据源前缀。
- 不跨场景强行合并身份。
- 无法证明同一车辆的空地样本不进入主 Re-ID 测试。
- full frame 可以只保留路径或索引，但必须保证 crop 可追溯。
- 所有脚本必须记录输入路径、输出路径、依赖版本、随机种子和配置文件。

## 6. 统一目录与文件命名

推荐根目录：

```text
tag_vr_dataset/
  README.md
  cvpair_index/
    cvpair_clean_index.jsonl
    cvpair_quality_notes.csv
  source_audit/
    source_audit.md
    source_schema_notes.md
    v2u4real_audit.md
    agc_drive_audit.md
    griffin_audit.md
    agvot_audit.md
  images/
    cvpair/
    v2u4real/
    agc_drive/
    griffin/
    agvot/
  full_frames/
    v2u4real/
    agc_drive/
    griffin/
    agvot/
  metadata/
    metadata.jsonl
    identity_links.jsonl
    split.json
  splits/
    train.json
    val.json
    test.json
  annotations/
    annotations_id.jsonl
    annotations_image.jsonl
  benchmarks/
    visual_a2g.jsonl
    visual_g2a.jsonl
    text_to_ground.jsonl
    text_to_uav.jsonl
    image_to_text.jsonl
    text_guided_cross_view.jsonl
    attribute_retrieval.jsonl
    cross_source_generalization.jsonl
  configs/
    conversion_config.yaml
    schema_mapping.yaml
    label_prompt.yaml
    split_config.yaml
  qa/
    qa_report.md
    low_quality_samples.csv
    unverified_associations.csv
    manual_review_queue.csv
```

### 6.1 图像命名

统一命名建议：

```text
{source_dataset}_{identity_scope}_{vehicle_id}_{view_source}_{sequence}_{frame:06d}_{instance:02d}.{ext}
```

示例：

```text
cvpair_global_0101_uav_s2_000010_00.jpg
v2u4real_global_000123_uav_s01_000120_00.jpg
v2u4real_global_000123_ground_vehicle_s01_000120_00.jpg
agc_drive_tracklet_000045_road_side_s02_000031_00.jpg
agvot_tracklet_000781_uav_s09_000240_00.jpg
```

字段说明：

- `source_dataset`: `cvpair`、`v2u4real`、`agc_drive`、`griffin`、`agvot` 或后续新增数据源名。
- `identity_scope`: `global`、`scene`、`tracklet`、`weak`。
- `vehicle_id`: 数据源内唯一 ID，推荐带源名前缀后进入 schema，例如 `v2u4real_global_000123`。
- `view_source`: `uav`、`ground_vehicle`、`road_side`、`ground_camera`。
- `sequence`: 原始序列、场景或片段编号。
- `frame`: 原始帧号或同步帧号。
- `instance`: 同帧多目标实例编号。

### 6.2 ID 命名

统一 `vehicle_id` 建议：

```text
{source_dataset}_{identity_scope}_{raw_vehicle_or_track_id}
```

示例：

```text
cvpair_global_0101
v2u4real_global_000123
agc_drive_tracklet_trk0092
agvot_tracklet_seq09_target01
```

不同数据源的 `raw_vehicle_id` 不得直接复用为全局 `vehicle_id`。

## 7. Metadata 与文本标注 schema

标注采用三层 JSONL：

- `metadata/metadata.jsonl`：每个 crop 一行，记录来源、路径、视角、bbox、时间、位姿、track/global ID、转换状态和 QA 状态。
- `metadata/identity_links.jsonl`：每个身份关联一行，记录同一车辆的空地样本、关联依据和可靠性。
- `annotations/annotations_id.jsonl`：每个 `vehicle_id` 一行，记录跨视角稳定身份属性和 ID 级自然语言描述。
- `annotations/annotations_image.jsonl`：每张图一行，记录实际可见属性、视角、遮挡、背景干扰、不确定字段和图像级描述。

### 7.1 metadata.jsonl 必选字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `dataset` | string | 固定为 `tag_vr` |
| `source_dataset` | string | 原始数据源 |
| `source_domain` | string | `real`、`simulated`、`mixed`、`unknown`，仅记录来源属性 |
| `source_task` | string | 原始任务，如 tracking、detection、cooperative_perception |
| `source_split` | string | 原始 split |
| `vehicle_id` | string | TAG-VR 统一车辆 ID |
| `raw_vehicle_id` | string/null | 原始车辆 ID |
| `track_id` | string/null | 原始 track ID |
| `global_id` | string/null | 原始 global ID |
| `identity_confidence` | string | 身份可靠性 |
| `association_method` | string | 关联方法 |
| `scene_id` | string | 场景 ID |
| `sequence_id` | string | 序列 ID |
| `timestamp` | number/null | 时间戳 |
| `frame_index` | integer/null | 帧号 |
| `platform_type` | string | 平台类型 |
| `platform_id` | string/null | 平台 ID |
| `sensor_id` | string/null | 传感器 ID |
| `camera_id` | string | `c0` 或 `c1` |
| `view_source` | string | 统一视角来源 |
| `image_path` | string | crop 相对路径 |
| `full_frame_path` | string/null | full frame 路径或索引 |
| `bbox_2d` | array/null | `[x1, y1, x2, y2]` |
| `bbox_3d` | object/null | 3D 框，若有 |
| `pose` | object/null | 车辆或目标位姿 |
| `category` | string | 原始或映射后的类别 |
| `target_size` | object | crop 或 bbox 尺寸 |
| `small_target` | boolean | 是否小目标 |
| `conversion_status` | string | 转换状态 |
| `qa_status` | string | QA 状态 |

示例：

```json
{
  "dataset": "tag_vr",
  "source_dataset": "v2u4real",
  "source_domain": "real",
  "source_task": "vehicle_to_uav_tracking",
  "source_split": "train",
  "scene_id": "scene_0007",
  "sequence_id": "seq_0007_02",
  "timestamp": 1716445212.42,
  "frame_index": 120,
  "platform_type": "uav",
  "platform_id": "uav_01",
  "sensor_id": "front_camera",
  "camera_id": "c1",
  "view_source": "uav",
  "image_path": "images/v2u4real/v2u4real_global_000123_uav_s01_000120_00.jpg",
  "full_frame_path": "full_frames/v2u4real/scene_0007/uav_01/front_camera/000120.jpg",
  "bbox_2d": [1240, 420, 1418, 552],
  "bbox_3d": {"center": [12.4, -3.1, 0.8], "size": [4.6, 1.8, 1.6], "yaw": 1.57},
  "track_id": "trk_0092",
  "global_id": "veh_000123",
  "raw_vehicle_id": "veh_000123",
  "vehicle_id": "v2u4real_global_000123",
  "identity_confidence": "global_id_verified",
  "association_method": "source_global_id",
  "pose": {"x": 12.4, "y": -3.1, "z": 0.8, "yaw": 1.57},
  "category": "car",
  "target_size": {"width_px": 178, "height_px": 132, "short_side_px": 132},
  "small_target": false,
  "conversion_status": "converted",
  "qa_status": "raw"
}
```

### 7.2 identity_links.jsonl 必选字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `vehicle_id` | string | TAG-VR 统一车辆 ID |
| `source_dataset` | string | 来源数据集 |
| `identity_scope` | string | `global`、`scene`、`tracklet`、`weak` |
| `identity_confidence` | string | 关联可靠性 |
| `association_method` | string | 关联方法 |
| `association_evidence` | array | 证据列表 |
| `uav_samples` | array | UAV 样本路径 |
| `ground_samples` | array | 地面样本路径 |
| `reviewer` | string/null | 人工复核人 |
| `qa_status` | string | QA 状态 |

示例：

```json
{
  "vehicle_id": "v2u4real_global_000123",
  "source_dataset": "v2u4real",
  "identity_scope": "global",
  "identity_confidence": "global_id_verified",
  "association_method": "source_global_id",
  "association_evidence": ["source_global_id", "timestamp_sync", "3d_spatial_consistency"],
  "uav_samples": ["images/v2u4real/v2u4real_global_000123_uav_s01_000120_00.jpg"],
  "ground_samples": ["images/v2u4real/v2u4real_global_000123_ground_vehicle_s01_000120_00.jpg"],
  "reviewer": null,
  "qa_status": "manual_checked"
}
```

### 7.3 annotations_id.jsonl 必选字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `vehicle_id` | string | TAG-VR 统一车辆 ID |
| `source_datasets` | array | 来源数据集列表 |
| `identity_scope` | string | 身份范围 |
| `identity_confidence` | string | 身份可靠性 |
| `description_zh` | string | 中文 ID 级稳定描述 |
| `description_en` | string | 英文 ID 级稳定描述 |
| `description_zh_variants` | array | 中文事实一致改写 |
| `description_en_variants` | array | 英文事实一致改写 |
| `caption_claims` | object | 按 canonical/paraphrase 保存原子 claim、claim type 和 fact IDs |
| `color` | string | 颜色 |
| `vehicle_type` | string | 粗粒度车型 |
| `body_profile` | string | 车身轮廓 |
| `roof_features` | array | 车顶结构 |
| `window_features` | array | 车窗特征 |
| `cargo_or_rear_structure` | string | 后部或厢体结构 |
| `special_marks` | array | 可见特殊标识 |
| `stable_attributes` | array | 跨视角稳定属性 |
| `uncertain_attributes` | array | 不确定属性 |
| `schema_qa_status` | string | schema QA 结果 |
| `fact_coverage_score` | number | ID stable fact 覆盖率 |
| `uncertainty_coverage` | number | 不确定项原因覆盖率 |
| `unsupported_claim_count` | integer | 无合法事实证据的 claim 数 |
| `claim_audit_status` | string | 最终 VLM B claim 审计状态 |
| `review_reasons` | array | 进入人工复核的原因 |
| `qa_status` | string | QA 状态 |

示例：

以下示例仅展示基础字段；正式 v4 输出还必须包含 variants、`caption_claims` 和 QA 字段，具体结构以 `configs/label_prompt.yaml` 为准。

```json
{
  "dataset": "tag_vr",
  "vehicle_id": "v2u4real_global_000123",
  "source_datasets": ["v2u4real"],
  "identity_scope": "global",
  "identity_confidence": "global_id_verified",
  "description_zh": "一辆白色轿车，车身轮廓较低，深色车窗，整体为普通乘用车外观。",
  "description_en": "A white sedan with a low body profile and dark windows.",
  "color": "white",
  "vehicle_type": "sedan",
  "body_profile": "low_body_profile",
  "roof_features": [],
  "window_features": ["dark_windows"],
  "cargo_or_rear_structure": "standard_trunk",
  "special_marks": [],
  "stable_attributes": ["white_body", "dark_windows", "low_body_profile"],
  "uncertain_attributes": [],
  "qa_status": "manual_checked"
}
```

### 7.4 annotations_image.jsonl 必选字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `image_path` | string | crop 相对路径 |
| `vehicle_id` | string | TAG-VR 统一车辆 ID |
| `source_dataset` | string | 数据源 |
| `camera_id` | string | `c0` 或 `c1` |
| `view_source` | string | 统一视角来源 |
| `platform_type` | string | 平台类型 |
| `orientation` | string | 车辆朝向或观测方向 |
| `visible_parts` | array | 可见部件 |
| `occlusion` | string | 遮挡程度 |
| `target_size` | object | 目标尺寸 |
| `small_target` | boolean | 是否小目标 |
| `scene_context` | string | 场景上下文 |
| `weather` | string/null | 天气，若可知 |
| `illumination` | string/null | 光照，若可知 |
| `background_distractors` | object | 背景干扰 |
| `description_zh` | string | 中文图像级描述 |
| `description_en` | string | 英文图像级描述 |
| `description_zh_variants` | array | 中文事实一致改写 |
| `description_en_variants` | array | 英文事实一致改写 |
| `caption_claims` | object | 按 canonical/paraphrase 保存原子 claim、claim type 和 fact IDs |
| `confidence` | number | 标注置信度 |
| `uncertain_fields` | array | 不确定字段 |
| `schema_qa_status` | string | schema QA 结果 |
| `fact_coverage_score` | number | 当前图高置信度可见事实覆盖率 |
| `uncertainty_coverage` | number | 当前图不确定项原因覆盖率 |
| `unsupported_claim_count` | integer | 无合法事实证据的 claim 数 |
| `claim_audit_status` | string | 最终 VLM B claim 审计状态 |
| `review_reasons` | array | 进入人工复核的原因 |
| `qa_status` | string | QA 状态 |

示例：

以下示例仅展示基础字段；正式 v4 输出还必须包含 variants、`caption_claims` 和 QA 字段，具体结构以 `configs/label_prompt.yaml` 为准。

```json
{
  "vehicle_id": "v2u4real_global_000123",
  "source_dataset": "v2u4real",
  "image_path": "images/v2u4real/v2u4real_global_000123_uav_s01_000120_00.jpg",
  "camera_id": "c1",
  "view_source": "uav",
  "platform_type": "uav",
  "orientation": "top_front",
  "visible_parts": ["roof", "windshield", "hood"],
  "occlusion": "partial",
  "target_size": {"width_px": 178, "height_px": 132, "short_side_px": 132},
  "small_target": false,
  "scene_context": "urban_road",
  "weather": "clear",
  "illumination": "daylight",
  "background_distractors": {"neighbor_vehicle_count": 3, "occluders": ["front_vehicle"], "complexity": "high"},
  "description_zh": "空中斜俯视下的白色轿车，可见车顶、前挡风玻璃和车头，前方车辆造成部分遮挡。",
  "description_en": "A white sedan viewed from an oblique aerial angle, with the roof, windshield, and hood visible and partial occlusion by a nearby vehicle.",
  "confidence": 0.86,
  "uncertain_fields": [],
  "qa_status": "manual_checked"
}
```

### 7.5 推荐枚举

- `source_dataset`: `cvpair`, `v2u4real`, `agc_drive`, `griffin`, `agvot`, `other`
- `source_domain`: `real`, `simulated`, `mixed`, `unknown`
- `platform_type`: `uav`, `ground_vehicle`, `road_side`, `ground_camera`, `unknown`
- `view_source`: `uav`, `ground_vehicle`, `road_side`, `ground_camera`
- `camera_id`: `c0`, `c1`
- `identity_scope`: `global`, `scene`, `tracklet`, `weak`
- `identity_confidence`: `global_id_verified`, `tracklet_verified`, `weak_association`, `unverified`
- `conversion_status`: `raw`, `converted`, `manual_review`, `rejected`, `held_for_release`
- `qa_status`: `raw`, `auto_labeled`, `manual_review`, `manual_checked`, `fixed`, `drop`
- `vehicle_type`: `sedan`, `suv`, `hatchback`, `van_minibus`, `pickup`, `light_truck`, `box_truck`, `bus`, `other`, `uncertain`
- `orientation`: `front`, `rear`, `left`, `right`, `front_left`, `front_right`, `rear_left`, `rear_right`, `top`, `top_front`, `top_rear`, `uncertain`
- `visible_parts`: `roof`, `hood`, `trunk`, `windshield`, `rear_window`, `side_windows`, `wheels`, `front_lights`, `rear_lights`, `side_body`, `cargo_box`, `roof_rack`, `sunroof`, `stripe`, `damage`
- `occlusion`: `none`, `slight`, `partial`, `heavy`, `uncertain`

## 8. 文本标注规则

文本标注分两层：

- ID 级稳定描述：每个车辆 ID 一条，强调跨视角稳定属性。
- 图像级属性描述：每张图一条，记录当前视角下实际可见的方向、部件、遮挡、背景和不确定项。

ID 级描述只写跨视角稳定车辆属性：

- 颜色
- 粗粒度车型
- 车身轮廓
- 车顶结构
- 车窗形态
- 厢体或后部结构
- 显著贴纸、条纹、损伤或其他可见特殊标识

图像级描述记录实际可见内容：

- 视角、方向、可见部件、遮挡、目标大小、背景干扰、天气、光照。
- 小目标、低光、重遮挡、颜色难辨时必须填写 `uncertain_fields`。
- 车辆类别不确定时使用 `vehicle_type="uncertain"`，不要为了构造检索标签强行归类。

禁止或谨慎项：

- 不自动生成品牌。
- 不自动生成具体型号。
- 不记录真实车牌号作为身份特征。
- 不描述图像中不可见的细节。
- 不把背景、停车位、道路纹理、邻车或车库柱体写成车辆身份特征。

VLM 辅助标注要求：

- 正式标注采用 `tag_vr_vehicle_annotation_v4_evidence_locked_claim_audit_2026-06-29`，不能把单次自由 caption 调用作为最终标注。
- VLM A 依次承担逐图结构化感知、同义词/冲突裁决、事实锁定 caption 生成和失败修复；VLM B 必须来自不同模型家族，只承担 claim-level 独立视觉审计。
- VLM A 的逐图感知不能生成 caption 或判断稳定性；`stable` 只能由程序按“至少 1 张 ground 支持、至少 1 张 UAV 支持、规范化 value 一致、无未解决 conflict”产生。
- caption 中每个正向事实必须引用一个合法 `fact_id`；事实表外观察必须通过 `new_observations` 回流感知和共识，不能直接写入 caption。
- 本地必须同时执行 schema、长度、事实覆盖和不确定项覆盖 QA。ID coverage 不低于 0.80，图像 coverage 不低于 0.85，uncertainty coverage 为 1.00，任一无证据事实直接失败。
- VLM B 在本地 QA 后审计，只看到原图、caption 和去除 VLM A confidence 的证据表。任一 claim 为 `contradicted` 或 `not_visible` 时，由 VLM A 修复、本地重检、VLM B 复审；仍失败则 `manual_review`。
- 必须保留 prompt、A/B 模型名和模型家族、调用时间、API usage、错误重试、程序共识结果、claim 审计和人工修改记录。
- 真实数据以图像可见内容为准。
- 扩展数据若提供 bbox、track ID、类别、视角、位姿或时间戳等元数据，应优先保留为结构化字段；文本描述仍以图像可见内容为准。
- 每个 ID 默认选择 2 张地面图和 2 张 UAV 图；逐图感知必须独立，避免地面图信息污染 UAV 判断。
- 中文和英文描述都需要保留，便于中文项目沟通和英文 CLIP/VLM baseline 复用。

## 9. Benchmark 任务与字段对应关系

`TAG-VR` 应保留传统 Re-ID 指标，同时扩展到文本增强检索任务。

| 任务 | 查询 | Gallery | 关键字段 | 可用样本 | 指标 |
| --- | --- | --- | --- | --- | --- |
| Visual a2g | UAV image | Ground images | `vehicle_id`, `camera_id`, `view_source`, `image_path` | CVPair `global_id_verified` 样本优先 | mAP, Rank-1/5/10 |
| Visual g2a | Ground image | UAV images | `vehicle_id`, `camera_id`, `view_source`, `image_path` | CVPair `global_id_verified` 样本优先 | mAP, Rank-1/5/10 |
| Text-to-ground | ID/text description | Ground images | `description_en`, `description_zh`, `vehicle_id`, `view_source` | 有 ID 级文本的 ground gallery | mAP, Recall@K |
| Text-to-UAV | ID/text description | UAV images | `description_en`, `description_zh`, `vehicle_id`, `view_source` | 有 ID 级文本的 UAV gallery | mAP, Recall@K |
| Image-to-text | Image | ID/text descriptions | `image_path`, `vehicle_id`, `annotations_id` | 有图像和 ID 文本的样本 | Recall@K |
| Text-guided cross-view retrieval | Image + text | Cross-view images | `image_path`, `description`, `vehicle_id`, `view_source` | CVPair 和可靠扩展样本 | mAP, Rank-K |
| Attribute retrieval | Color/type/orientation attributes | Images | `color`, `vehicle_type`, `orientation`, `visible_parts` | 图像级属性完整样本 | Recall@K, attribute accuracy |
| Tracklet retrieval | UAV or ground tracklet | Cross-view tracklets | `track_id`, `vehicle_id`, `identity_confidence` | `tracklet_verified` 及以上 | mAP, Rank-K |
| Cross-source generalization | One source image/text | Held-out source images | `source_dataset`, `vehicle_id`, `description`, `image_path` | 按 source 留出 | mAP, Rank-K, Recall@K |

使用边界：

- 主实例级 Re-ID 测试集只使用 `global_id_verified`。
- `tracklet_verified` 可用于 tracklet retrieval 或训练增强，但报告时需单列。
- `weak_association` 不进入主测试集，只用于辅助训练、人工复核或附录分析。
- `unverified` 不进入 benchmark。
- baseline 实验必须分别报告 `a2g` 和 `g2a`，不要只给合并结果。

## 10. 质量检查和验收标准

### 10.1 脚本检查

交付前必须完成脚本检查：

- 所有 JSONL 文件一行一个合法 JSON object，可逐行解析。
- `metadata.jsonl` 中每个 `image_path` 均可读取。
- `annotations_image.jsonl` 中每个 `image_path` 能在 `metadata.jsonl` 找到。
- `annotations_id.jsonl` 中每个 `vehicle_id` 至少有一张图像，除非明确标记为待补。
- `identity_links.jsonl` 中的样本路径必须存在于 `metadata.jsonl`。
- `vehicle_id` 不跨 source 误复用。
- train、val、test 无 `vehicle_id` 泄漏。
- 主测试集不包含 `weak_association` 或 `unverified`。
- `qa_status="drop"` 的样本不进入 benchmark 索引。
- `.jpg`、`.png` 等扩展名与实际文件格式不一致时必须记录。

### 10.2 人工抽查

最低抽查建议：

- CVPair 每个 split 抽查至少 30 个 ID；每个 ID 至少看 1 张 `c0` 和 1 张 `c1`。
- 每个扩展数据源抽查至少 30 个车辆关联；不足 30 个则全量抽查。
- 对 `weak_association`、小目标、重遮挡、多车混淆样本提高抽查比例。
- 对 VLM 自动标注样本抽查颜色、车型、可见部件、遮挡和背景泄漏。
- 检查 `schema_qa.jsonl`、`caption_length_qa.jsonl`、`fact_coverage_qa.jsonl`、`claim_audit.jsonl` 和 `claim_audit_summary.jsonl` 均有记录。
- query/gallery 全量人工复核；训练 ID 按固定 seed 至少抽取 10%，并对低置信度、冲突、小目标和重遮挡样本提高比例。

人工抽查通过率建议不低于 95%。未通过样本标记为 `manual_review`、`fixed` 或 `drop`。

### 10.3 低质样本处理

应标记为 `drop` 或不进入主数据的情况：

- 非车辆目标。
- 无法确认空地视角是否对应同一目标。
- crop 中主目标严重截断、完全不可辨认或被邻近目标混淆。
- 原始 bbox、timestamp、sensor ID 或路径无法追溯。
- 文本描述出现不可见品牌、型号、车牌号。
- 文本把背景、停车位、道路纹理、柱体或邻车布局写成身份属性。
- 身份关联只有外观相似但没有时间、空间、track 或人工复核证据。

### 10.4 QA report 必含内容

`qa/qa_report.md` 至少包含：

- 数据源列表、版本、下载日期和许可状态。
- CVPair 图像数、唯一 basename 数、ID 数、split 数、重复训练目录说明。
- 每个扩展数据源的原始序列数、可用序列数、车辆 crop 数、空地配对数。
- `global_id_verified`、`tracklet_verified`、`weak_association`、`unverified` 数量。
- ID 级文本覆盖率和图像级文本覆盖率。
- 车辆类型、颜色、视角、遮挡、小目标分布。
- `manual_review`、`fixed`、`drop` 数量和原因。
- 已知风险：背景泄漏、域差、身份关联误差、VLM 幻觉、许可限制。
- 后续需要补充或不能公开的内容。

## 11. 工作节奏建议

| 阶段 | 时间估算 | 关键检查点 |
| --- | ---: | --- |
| 需求对齐和 schema 确认 | 0.5 天 | 确认数据源、字段、目录和验收口径 |
| CVPair 清洗索引 | 1 天 | 生成干净索引和质量问题表 |
| 扩展数据源 audit | 1-2 天 | 每个候选源都有下载、许可和 schema 记录 |
| 小样本转换 smoke test | 2 天 | 每个可用源转换 3-5 个 sequence |
| 身份可靠性复核 | 2-4 天 | 区分 global、tracklet、weak、unverified |
| 文本标注和 QA | 3-5 天 | 完成 ID 级和图像级标注抽查 |
| Benchmark split 与交付 | 1 天 | README、QA report、schema mapping、任务索引完整 |

建议先完成 CVPair 和一个可用扩展数据源的端到端 smoke test，再扩展到更多数据源。

## 12. 最终交付清单

最终交付应包含：

- `README.md`：数据说明、目录说明、许可边界、字段解释、使用示例。
- `cvpair_index/cvpair_clean_index.jsonl`：CVPair/CVnet 干净索引。
- `cvpair_index/cvpair_quality_notes.csv`：真实数据异常、格式问题和抽查记录。
- `source_audit/`：每个扩展数据源的下载状态、许可、schema 和限制说明。
- `images/`：转换后的车辆 crop。
- `full_frames/`：可追溯的原始帧路径、索引或必要副本。
- `metadata/metadata.jsonl`：每个 crop 一行完整元数据。
- `metadata/identity_links.jsonl`：身份关联关系和可靠性证据。
- `metadata/split.json`：训练、验证、测试和任务划分。
- `splits/train.json`、`splits/val.json`、`splits/test.json`：按 ID、source、scene、tracklet 防泄漏后的全局划分。
- `annotations/annotations_id.jsonl`：每个 ID 或 tracklet 一行稳定文本描述。
- `annotations/annotations_image.jsonl`：每张图一行图像级属性和描述。
- `benchmarks/visual_a2g.jsonl`：visual a2g 任务索引。
- `benchmarks/visual_g2a.jsonl`：visual g2a 任务索引。
- `benchmarks/text_to_ground.jsonl`：text-to-ground 任务索引。
- `benchmarks/text_to_uav.jsonl`：text-to-UAV 任务索引。
- `benchmarks/image_to_text.jsonl`：image-to-text 任务索引。
- `benchmarks/text_guided_cross_view.jsonl`：文本引导跨视角检索任务索引。
- `benchmarks/attribute_retrieval.jsonl`：属性检索任务索引。
- `benchmarks/cross_source_generalization.jsonl`：跨数据源泛化任务索引。
- `configs/schema_mapping.yaml`：原始字段到 TAG-VR 字段的映射。
- `configs/conversion_config.yaml`：筛选类别、crop 参数、split 规则和随机种子。
- `configs/label_prompt.yaml`：VLM 标注 prompt、字段约束和禁止项。
- `qa/qa_report.md`：规模统计、身份可靠性统计、车辆类别统计、文本覆盖率和已知问题。
- `qa/low_quality_samples.csv`：低质、待复核或丢弃样本清单。
- `qa/unverified_associations.csv`：弱关联和未验证关联样本清单。
- `qa/manual_review_queue.csv`：人工复核队列。
