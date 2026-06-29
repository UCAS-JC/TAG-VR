# TAG-VR 文本标注方法改进方案

生成日期：2026-06-29  
方法版本：`tag_vr_vehicle_annotation_v4_evidence_locked_claim_audit_2026-06-29`

## 1. 参考论文与可迁移方法

| 论文 | 原始数据标注方法 | 可迁移到 TAG-VR 的部分 | 不直接照搬的部分 |
| --- | --- | --- | --- |
| Text-based Aerial-Ground Person Retrieval / TAG-PEDES | MLLM 的 prompt-based、template-based、attribute-based 三种生成策略；每张图生成两条描述；测试集由 6 名合格标注者修订；用 Match、Contradictory、Hallucinatory、Vacuous 四类评估图文质量 | 双描述、属性先验、模板池小样本筛选、四类语义 QA、测试集人工复核 | 行人服饰模板必须替换为车辆属性；固定模板不能成为正式 caption 的唯一来源 |
| Cross-modal Fuzzy Alignment Network / AERI-PEDES | CoT 三阶段：结构化视觉解析并记录属性/证据/置信度，生成初稿，再由视觉审计模型修正遗漏和幻觉；训练文本自动生成，测试文本人工标注 | `perception -> caption -> audit/refine` 多阶段标注；保存字段证据、置信度、修订项；测试集强制人工复核 | 不保存模型私有推理链，只保存可审计的属性证据、问题清单和修订记录 |
| AEA-FIRM / TBAPR | 从地面图生成文本并关联到同 ID 空中图；CLIP 低置信度过滤后重生成；跨模型投票和同 ID 多图投票提取视角不变属性；属性文本化并构造同属性类别困难负样本 | 地面图作为细节丰富的语义锚点、跨视角属性投票、相似度/语义分数门控、属性文本化、同属性困难负样本 | 不能把同一地面 caption 直接复制为 UAV 图像级 caption；只能把跨视角确认的属性写入 ID 级描述 |

## 2. TAG-VR-CAP v4 定稿流程

### 0. 图像质量计算和代表图选择

- 每个 ID 默认选择 4 张图：2 张 `ground_camera` 和 2 张 `uav`。
- 优先非小目标、短边更大、主车清晰且路径可追溯的 crop。
- 小目标、遮挡和强干扰样本保留，但质量字段必须进入后续不确定性判断。

### 1. VLM A 逐图结构化感知

- 同一请求可输入 4 张图，但每张图必须独立输出，不能跨图补全不可见属性。
- 本阶段禁止生成 caption，禁止判断 `stable`。
- 每条记录必须包含 `image_path`、`attribute`、`value`、`confidence`、`visibility`、`visual_evidence`、`source_images` 和 `quality_notes`。
- `source_images` 只能包含当前图；`visible_part` 按部件拆成独立记录。
- `visibility` 取值为 `visible`、`partial`、`not_visible`、`uncertain`、`not_applicable`。

### 2. 规则程序与 VLM A 共同形成跨视角共识

VLM A 只能处理同义值、解释冲突或建议降级为 `uncertain`。最终规则归程序所有：

```text
stable = ground_support_count >= 1
         AND uav_support_count >= 1
         AND normalized_value 一致
         AND 不存在未解决 conflict
```

程序拒绝 VLM A 对已知规范值的任意重映射，也忽略其直接 `stable` 建议。非跨视角稳定属性，如 `visible_part`、`orientation` 和 `occlusion`，只进入 `view_specific`。

### 3. VLM A 基于事实表生成中英文双描述

- 结构化事实表是唯一权威输入，图片只用于核对措辞和当前视角。
- 每个正向 `caption_claim` 必须是原子事实，并引用且仅引用一个合法 `fact_id`。
- ID caption 只能引用 `stable` facts；图像 caption 只能引用当前图的高置信度 `visible/partial` facts。
- 如果图像出现事实表外的新属性，不得写入 caption，必须返回 `new_observations`，并重新进入阶段 1、2、3；默认最多回流 1 次。
- 每个 ID 和图像均生成 `canonical` 与 `natural_paraphrase_1`，两者不得改变事实集合。

长度标准保持为：ID 中文 60-100 汉字、英文 35-60 words；图像中文 50-90 汉字、英文 30-55 words；均为 1-2 句。

### 4. 本地 schema、长度和事实覆盖 QA

本地程序同时执行：

```text
ID coverage = caption 已描述的 stable fact 数 / 全部适用 stable fact 数
Image coverage = caption 已描述的高置信度 visible/partial fact 数 / 当前图全部适用高置信度 visible/partial fact 数
uncertainty coverage = 已解释原因的不确定属性数 / 应解释的不确定属性数
```

门槛：ID `>= 0.80`，图像 `>= 0.85`，不确定项 `= 1.00`。`not_visible` 和 `not_applicable` 不进入分母；`uncertain` 和 `conflict` 不得作为正向事实。任何未知 `fact_id`、claim 与 fact 不一致或无证据事实均直接失败。

可见部件规则：确认至少 3 个部件时 caption 至少覆盖 3 个；少于 3 个时必须覆盖全部可确认部件，并解释为何只能确认较少部件。

### 5. 不同模型家族的 VLM B 做 claim-level 独立视觉审计

- VLM B 必须与 VLM A 来自不同模型家族。
- VLM B 看到原图、最终 caption 和去除 confidence 的证据表；不看到 VLM A 的 caption claim plan，避免锚定。
- VLM B 自行拆解全部 claim，逐条输出 `supported`、`contradicted` 或 `not_visible`、`source_images` 和 `correction`。
- 任一 claim 非 `supported`、任一 variant 未完整审计，均判失败。

### 6-8. 失败修复与复审

若第一轮本地 QA 或 VLM B 审计失败：

1. VLM A 仅根据事实表、本地 QA 失败项和 VLM B 问题清单修复 caption。
2. 本地重新执行 schema、长度、事实覆盖和不确定项覆盖 QA。
3. VLM B 对修复结果执行第二轮独立 claim-level 审计。
4. 本地 QA 或第二轮审计仍失败时，标记 `manual_review`。

### 9. 人工复核

- query/gallery 评测 ID 和图像全部人工复核。
- 训练集按固定 seed 对 ID 做可复现抽样，默认抽取 10%。
- 低置信度、冲突、小目标、重遮挡和多车混淆样本可提高抽样比例。
- 人工修订保留 `before`、`after`、`reviewer`、`reason` 和时间戳。

## 3. 输出产物

```text
annotations/<run>/
  annotations_id.jsonl
  annotations_image.jsonl
  annotation_evidence.jsonl
  semantic_qa.jsonl
  schema_qa.jsonl
  caption_length_qa.jsonl
  fact_coverage_qa.jsonl
  claim_audit.jsonl
  claim_audit_summary.jsonl
  raw_responses.jsonl
  run_manifest.json
  annotation_smoke_test_report.md
```

人工复核和困难负样本仍建议单独保存：

```text
annotations/hard_negatives/
  attribute_hard_negatives.jsonl
qa/
  manual_review_queue.csv
  human_revision_log.jsonl
```

## 4. 自动通过门槛

- 每个 `stable` fact 严格满足程序公式，VLM A 无权绕过规则升级。
- 每个 canonical 和 paraphrase 同时通过 schema、长度、事实覆盖、不确定项覆盖和无证据事实检查。
- 第一轮或修复后的最终 VLM B claim 审计全部为 `supported`。
- 任何品牌、具体型号、真实车牌、不可见细节或背景身份捷径均失败。
- query/gallery 即使自动 QA 全部通过，仍必须人工复核。

## 5. 与现有方法的关系

v2 解决了 caption 偏短和长度 QA；v3 增加了多图共识和视觉审计。v4 进一步把感知、共识、生成和审计拆分，将 `stable` 决策权收回程序，引入 `fact_id`、coverage、uncertainty coverage、异构 VLM B claim 审计以及失败后的修复复审闭环。
