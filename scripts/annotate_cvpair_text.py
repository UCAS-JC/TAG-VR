#!/usr/bin/env python3
"""Run a small CVPair VLM text-annotation job for TAG-VR.

The script reads the cleaned `metadata/metadata.jsonl` produced by
`build_cvpair_dataset.py`, samples one or more vehicle IDs with both aerial and
ground views, calls an OpenAI-compatible vision-language API, and writes
auditable ID-level and image-level annotation JSONL files.
"""

from __future__ import annotations

import argparse
import base64
import copy
from collections import defaultdict
from datetime import datetime, timezone
import colorsys
import hashlib
import io
import json
import mimetypes
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Optional
import urllib.error
import urllib.request

from PIL import Image


CONTROLLED_ENUMS = {
    "vehicle_type": [
        "sedan",
        "suv",
        "hatchback",
        "van_minibus",
        "pickup",
        "light_truck",
        "box_truck",
        "bus",
        "other",
        "uncertain",
    ],
    "orientation": [
        "front",
        "rear",
        "left",
        "right",
        "front_left",
        "front_right",
        "rear_left",
        "rear_right",
        "top",
        "top_front",
        "top_rear",
        "uncertain",
    ],
    "visible_parts": [
        "roof",
        "hood",
        "trunk",
        "windshield",
        "rear_window",
        "side_windows",
        "wheels",
        "front_lights",
        "rear_lights",
        "side_body",
        "cargo_box",
        "roof_rack",
        "sunroof",
        "stripe",
        "damage",
    ],
    "occlusion": ["none", "slight", "partial", "heavy", "uncertain"],
    "qa_status": ["auto_labeled", "manual_review", "manual_checked", "fixed", "drop"],
}

SEMANTIC_QA_CATEGORIES = ["match", "contradictory", "hallucinatory", "vacuous"]

STABLE_VEHICLE_ATTRIBUTES = [
    "color",
    "vehicle_type",
    "body_profile",
    "body_proportions",
    "roof_features",
    "window_features",
    "cargo_or_rear_structure",
    "special_marks",
]


CAPTION_LENGTH_RULES = {
    "id": {
        "zh_min": 60,
        "zh_max": 100,
        "en_min": 35,
        "en_max": 60,
        "sentence_min": 1,
        "sentence_max": 2,
    },
    "image": {
        "zh_min": 50,
        "zh_max": 90,
        "en_min": 30,
        "en_max": 55,
        "sentence_min": 1,
        "sentence_max": 2,
    },
}


ANNOTATION_METHOD_VERSION = "tag_vr_vehicle_annotation_v4_evidence_locked_claim_audit_2026-06-29"
PROMPT_VERSION = "tag_vr_cvpair_caption_v4_staged_evidence_coverage_2026-06-29"

EVIDENCE_CONFIDENCE_THRESHOLD = 0.75
ID_COVERAGE_THRESHOLD = 0.80
IMAGE_COVERAGE_THRESHOLD = 0.85
UNCERTAINTY_COVERAGE_THRESHOLD = 1.0
POSITIVE_VISIBILITY_VALUES = {"visible", "partial"}
EXCLUDED_VISIBILITY_VALUES = {"not_visible", "not_applicable"}


SYSTEM_PROMPT = """你是 TAG-VR / Text-enhanced Aerial-Ground Vehicle Retrieval 项目的车辆文本标注助手。
你的任务是基于输入车辆 crop 的可见内容，输出可解析 JSON。必须遵守：
1. 只描述图像中可见的车辆属性，不猜品牌、具体型号和真实车牌。
2. ID 级描述只写跨视角稳定车辆属性；背景、停车位、道路纹理、柱体和邻车不能写成身份特征。
3. 图像级描述可记录当前视角、可见部件、遮挡、背景干扰和不确定字段。
4. 小目标、遮挡、颜色难辨、车型难辨时使用 uncertain，并把字段写入 uncertain_fields。
5. 同时输出中文 description_zh 和英文 description_en。
6. ID 级 description_zh 为 60-100 个中文汉字，description_en 为 35-60 个英文单词；各用信息密度高的 1-2 句。
7. 图像级 description_zh 为 50-90 个中文汉字，description_en 为 30-55 个英文单词；各用信息密度高的 1-2 句。
8. 低于上述长度下限的 caption 视为无效，即使语义正确也必须扩写；返回前自行检查字数、词数和句数。
9. ID 级描述优先覆盖颜色、粗粒度车型、车身轮廓与比例、车窗/挡风玻璃、车顶或后部结构、特殊标记、跨视角稳定线索，以及不确定属性和原因。
10. 图像级描述先写主车辆本体，再写当前视角、朝向、可见部件、遮挡、邻车或背景干扰，最后说明不确定字段及原因。
11. 不要用“车身特征可见”等空泛短句凑字段；描述应随图像实际属性变化，中文与英文语义对齐，不逐字段机械罗列，也不要为了长度添加无关背景。
12. 不要把同一句 caption 复制给 ID、地面图和 UAV 图；ID 级、地面图像级、UAV 图像级必须各自体现对应层级和视角。
13. 先按图像记录结构化属性、可见证据和置信度，再通过同 ID 多图形成 stable、view_specific、conflict、uncertain 四类跨视角共识。
14. 地面图可以作为细节丰富的语义锚点，但未经 UAV 图或其他同 ID 图像支持的属性不能进入 ID 级稳定描述；地面 caption 不得直接复制为 UAV 图像级 caption。
15. canonical caption 和 natural paraphrase 必须基于同一属性事实，paraphrase 只改变表达，不得新增细节。
16. 语义审计使用 match、contradictory、hallucinatory、vacuous 四类；出现冲突、幻觉或空泛描述时必须修正并记录 correction。
17. 只输出可复核的 visual_evidence、confidence、field_issues 和 corrections，不输出模型私有推理链。
18. 不要输出 Markdown、解释文字或代码块，只输出一个 JSON object。
"""


AUDIT_SYSTEM_PROMPT = """你是 TAG-VR 车辆图文标注的独立视觉审计员。
请重新查看所有输入图像，逐项检查初始标注是否与可见证据一致，并返回精修后的 JSON。
质量类别只能是 match、contradictory、hallucinatory、vacuous：
- match：描述准确、属性充分且没有不可见事实；
- contradictory：至少一个属性与图像直接冲突；
- hallucinatory：包含图中不存在或不可见的细节；
- vacuous：描述空泛，缺少有检索价值的车辆属性。
地面图可以提供更丰富细节，但不能把未被 UAV 或其他同 ID 图支持的属性写成 ID 级稳定线索。
canonical 与 natural paraphrase 必须语义一致，paraphrase 不得新增事实。
ID 级 caption 必须满足中文 60-100 个汉字、英文 35-60 个单词；图像级 caption 必须满足中文 50-90 个汉字、英文 30-55 个单词。短于下限属于 vacuous，必须扩写后再返回。
不要把同一句 caption 复制给 ID、地面图和 UAV 图；每条图像级 caption 必须体现当前视角可见内容、遮挡或背景干扰。
请直接修正错误和遗漏，记录简短 field_issues、corrections、omitted_attributes，不输出私有推理链。
不要输出 Markdown 或解释文字，只输出一个 JSON object。
"""


CAPTION_REPAIR_SYSTEM_PROMPT = """你是 TAG-VR caption length QA 修复器。
你只能基于已有结构化视觉证据、跨视角共识、图像级属性和语义审计结果扩写 caption，不得新增品牌、具体型号、车牌或不可见细节。
修复目标：
- ID 级 description_zh 与每个中文 paraphrase 为 60-100 个中文汉字，description_en 与每个英文 paraphrase 为 35-60 个英文单词。
- 图像级 description_zh 与每个中文 paraphrase 为 50-90 个中文汉字，description_en 与每个英文 paraphrase 为 30-55 个英文单词。
- 每条 caption 为信息密度高的 1-2 句，不写字段清单，不写空泛模板，不把背景作为身份捷径。
- ID 级只写跨视角稳定车辆属性；图像级写当前视角可见部件、遮挡、邻车/背景干扰和不确定原因。
返回前必须自行检查每条 canonical 和 paraphrase 的长度。不要输出 Markdown 或解释文字，只输出一个 JSON object。
"""


PERCEPTION_SYSTEM_PROMPT = """你是 TAG-VR 的逐图车辆结构化感知模型 VLM A。
每张图必须独立判断，不能参考同 ID 其他图像来补全当前图不可见的属性。
本阶段只输出逐图结构化证据，不生成 caption，不做跨视角稳定性判断。
每条属性必须包含 attribute、value、confidence、visibility、visual_evidence、source_images；source_images 只能包含当前 image_path。
visible_part 必须按部件拆成多条属性记录，不返回一个不可追踪的部件列表。
不可见写 not_visible，不适用写 not_applicable，证据不足写 uncertain；不要猜品牌、具体型号或车牌。
只输出 JSON，不输出私有推理链、Markdown 或额外解释。
"""


CONSENSUS_SYSTEM_PROMPT = """你是 TAG-VR 的跨视角属性裁决模型 VLM A。
输入包含确定性程序产生的候选共识和逐图证据。本阶段不得生成 caption，也不得直接决定 stable。
你只能：提出同义值规范化建议；解释冲突；建议将属性降级为 uncertain。
你不得绕过程序规则把属性升级为 stable。最终 stable 只由程序按 ground/UAV 支持数、规范化 value 一致性和未解决冲突决定。
只输出 JSON，不输出私有推理链、Markdown 或额外解释。
"""


FACT_LOCKED_CAPTION_SYSTEM_PROMPT = """你是 TAG-VR 的事实锁定 caption 生成模型 VLM A。
结构化事实表是唯一权威事实来源，图像只用于核对表达是否与当前视角一致。
caption 中每个正向事实都必须在 caption_claims 中引用有效 fact_ids；不得绕过事实表增加新属性。
如果图像中发现事实表未包含的新属性，不得写入 caption，必须放入 new_observations，等待重新进入感知和共识阶段。
ID caption 只使用 stable facts；图像 caption 只使用当前图像的高置信度 visible/partial facts。
不确定项必须以 uncertainty claim 给出可见原因。canonical 和 paraphrase 必须事实一致。
只输出 JSON，不输出私有推理链、Markdown 或额外解释。
"""


CLAIM_AUDIT_SYSTEM_PROMPT = """你是与 VLM A 不同模型家族的独立视觉审计模型 VLM B。
你会看到原图、最终 caption 和已去除置信度的证据表。不要推测 VLM A 的意图，也不要受其自评置信度锚定。
请自行拆解每条 caption 的全部可验证 claim，并逐条标记 supported、contradicted 或 not_visible。
每条 claim 必须返回 source_images 和必要的 correction。背景只可作为图像上下文，不能作为 ID 身份属性。
不得只返回整体分数；必须覆盖 canonical 和每个 paraphrase 的 claim-level 审计。
只输出 JSON，不输出私有推理链、Markdown 或额外解释。
"""


AUDIT_REPAIR_SYSTEM_PROMPT = """你是 TAG-VR 的事实锁定修复模型 VLM A。
根据本地 schema/长度/覆盖 QA 失败项和 VLM B 的 claim-level 问题清单修复 caption。
只能使用给定事实表中的 fact_ids；不得增加新属性，不得修改证据表或程序共识。
修复后必须保留完整 annotation_id、annotations_image、caption_claims 和中英文 canonical/paraphrase。
只输出 JSON，不输出私有推理链、Markdown 或额外解释。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate CVPair images with a VLM.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("tag_vr_dataset"),
        help="TAG-VR dataset root produced by build_cvpair_dataset.py.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional metadata.jsonl path. Defaults to dataset-root/metadata/metadata.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to dataset-root/annotations/cvpair_smoke_test.",
    )
    parser.add_argument(
        "--vehicle-id",
        action="append",
        default=[],
        help="Specific vehicle_id to annotate. May be passed multiple times.",
    )
    parser.add_argument("--limit-ids", type=int, default=1, help="Number of IDs to annotate.")
    parser.add_argument(
        "--images-per-id",
        type=int,
        default=4,
        help="Maximum images sent for each vehicle ID.",
    )
    parser.add_argument(
        "--require-both-views",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require sampled IDs to include both uav and ground_camera views.",
    )
    parser.add_argument(
        "--model",
        "--model-a",
        dest="model",
        default=os.environ.get("TAG_VR_VLM_A_MODEL")
        or os.environ.get("TAG_VR_VLM_MODEL", "gpt-4o-mini"),
        help="OpenAI-compatible VLM A model used for perception, consensus, captioning, and repair.",
    )
    parser.add_argument(
        "--model-b",
        default=os.environ.get("TAG_VR_VLM_B_MODEL"),
        help="Independent VLM B model used for claim-level visual audit.",
    )
    parser.add_argument(
        "--model-a-family",
        default=os.environ.get("TAG_VR_VLM_A_FAMILY"),
        help="Optional explicit model-family label for VLM A.",
    )
    parser.add_argument(
        "--model-b-family",
        default=os.environ.get("TAG_VR_VLM_B_FAMILY"),
        help="Optional explicit model-family label for VLM B; must differ from VLM A.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("TURINGAI_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://turingai.plus/v1",
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api-key-env",
        default="TURINGAI_API_KEY",
        help="Environment variable containing the VLM A API key.",
    )
    parser.add_argument(
        "--audit-base-url",
        default=os.environ.get("TAG_VR_VLM_B_BASE_URL"),
        help="OpenAI-compatible VLM B base URL. Defaults to --base-url.",
    )
    parser.add_argument(
        "--audit-api-key-env",
        default="TAG_VR_VLM_B_API_KEY",
        help="Environment variable containing the VLM B API key; falls back to the VLM A key.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=1024,
        help="Resize images for API payload so the longest side is at most this many pixels.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="JPEG quality used after resizing images for API calls.",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--backend",
        choices=("heuristic", "api", "mock"),
        default="heuristic",
        help="Annotation backend. heuristic is local and does not upload images.",
    )
    parser.add_argument(
        "--pipeline",
        choices=("staged_claim_audit", "single_pass", "cot_audit"),
        default="staged_claim_audit",
        help="staged_claim_audit is the v4 evidence-locked A/B pipeline; older modes are retained for provenance.",
    )
    parser.add_argument(
        "--caption-variants",
        type=int,
        choices=(1, 2),
        default=2,
        help="Total canonical plus natural-paraphrase captions per ID/image record.",
    )
    parser.add_argument(
        "--semantic-match-threshold",
        type=float,
        default=0.75,
        help="Minimum audited image-text match score for automatic training-set acceptance.",
    )
    parser.add_argument(
        "--evidence-confidence-threshold",
        type=float,
        default=EVIDENCE_CONFIDENCE_THRESHOLD,
        help="Minimum confidence for positive visible/partial evidence to enter coverage denominators.",
    )
    parser.add_argument(
        "--id-coverage-threshold",
        type=float,
        default=ID_COVERAGE_THRESHOLD,
        help="Minimum ID-caption stable-fact coverage.",
    )
    parser.add_argument(
        "--image-coverage-threshold",
        type=float,
        default=IMAGE_COVERAGE_THRESHOLD,
        help="Minimum image-caption visible-fact coverage.",
    )
    parser.add_argument(
        "--uncertainty-coverage-threshold",
        type=float,
        default=UNCERTAINTY_COVERAGE_THRESHOLD,
        help="Minimum fraction of required uncertain fields whose reasons are explained.",
    )
    parser.add_argument(
        "--max-new-observation-rounds",
        type=int,
        default=1,
        help="Maximum caption new_observation loops back through perception and consensus.",
    )
    parser.add_argument(
        "--human-review-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force query/gallery evaluation identities into manual review.",
    )
    parser.add_argument(
        "--train-review-rate",
        type=float,
        default=0.10,
        help="Deterministic fraction of training identities routed to human review.",
    )
    parser.add_argument(
        "--review-sampling-seed",
        type=int,
        default=20260629,
        help="Seed used for reproducible training-identity human-review sampling.",
    )
    parser.add_argument(
        "--repair-caption-length",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Legacy single_pass/cot_audit text-only repair; v4 uses the claim-audit repair loop.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Backward-compatible alias for --backend mock.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files in output-dir.",
    )
    return parser.parse_args()


def infer_model_family(model: Optional[str]) -> Optional[str]:
    value = (model or "").lower()
    if not value:
        return None
    families = {
        "openai": ("gpt-", "o1", "o3", "o4"),
        "qwen": ("qwen",),
        "anthropic": ("claude",),
        "google": ("gemini", "paligemma"),
        "zhipu": ("glm", "cogvlm"),
        "meta": ("llama",),
        "mistral": ("mistral", "pixtral"),
        "internvl": ("internvl",),
        "deepseek": ("deepseek",),
    }
    for family, markers in families.items():
        if any(marker in value for marker in markers):
            return family
    return None


def validate_numeric_args(args: argparse.Namespace) -> None:
    bounded = {
        "evidence_confidence_threshold": args.evidence_confidence_threshold,
        "id_coverage_threshold": args.id_coverage_threshold,
        "image_coverage_threshold": args.image_coverage_threshold,
        "uncertainty_coverage_threshold": args.uncertainty_coverage_threshold,
        "train_review_rate": args.train_review_rate,
    }
    for name, value in bounded.items():
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be between 0 and 1.")
    if args.max_new_observation_rounds < 0:
        raise SystemExit("--max-new-observation-rounds must be non-negative.")


def validate_v4_api_configuration(args: argparse.Namespace) -> tuple[str, str, str]:
    if not args.model_b:
        raise SystemExit(
            "staged_claim_audit requires --model-b or TAG_VR_VLM_B_MODEL from a different model family."
        )
    family_a = args.model_a_family or infer_model_family(args.model)
    family_b = args.model_b_family or infer_model_family(args.model_b)
    if not family_a or not family_b:
        raise SystemExit(
            "Could not infer both model families; pass --model-a-family and --model-b-family explicitly."
        )
    if family_a.lower() == family_b.lower():
        raise SystemExit(
            f"VLM A and VLM B must use different model families, got {family_a!r} for both."
        )
    return family_a.lower(), family_b.lower(), args.audit_base_url or args.base_url


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def mime_type_for(path: Path) -> str:
    guessed = mimetypes.guess_type(path.name)[0]
    if guessed:
        return guessed
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


def image_to_data_url(path: Path, max_side: int, jpeg_quality: int) -> str:
    if max_side > 0:
        with Image.open(path) as img:
            img.thumbnail((max_side, max_side))
            if img.mode not in {"RGB", "L"}:
                img = img.convert("RGB")
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
            data = buffer.getvalue()
        return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")

    data = path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type_for(path)};base64,{encoded}"


def group_by_vehicle(metadata: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metadata:
        if row.get("source_dataset") != "cvpair":
            continue
        if row.get("qa_status") == "drop":
            continue
        grouped[row["vehicle_id"]].append(row)
    return dict(grouped)


def select_representative_images(
    rows: list[dict[str, Any]], images_per_id: int
) -> list[dict[str, Any]]:
    by_view: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_view[row.get("view_source", "unknown")].append(row)

    def quality_key(row: dict[str, Any]) -> tuple[Any, ...]:
        target_size = row.get("target_size") or {}
        short_side = int(target_size.get("short_side_px") or 0)
        return (
            bool(row.get("small_target")),
            -short_side,
            row.get("image_path", ""),
        )

    selected: list[dict[str, Any]] = []
    balanced_quota = images_per_id // 2
    for view in ("ground_camera", "uav"):
        view_rows = sorted(by_view.get(view, []), key=quality_key)
        selected.extend(view_rows[:balanced_quota])

    if images_per_id % 2 and by_view:
        preferred_view = max(
            ("ground_camera", "uav"),
            key=lambda view: len(by_view.get(view, [])),
        )
        existing = {row["image_path"] for row in selected}
        extra = [
            row
            for row in sorted(by_view.get(preferred_view, []), key=quality_key)
            if row["image_path"] not in existing
        ]
        selected.extend(extra[:1])

    if len(selected) < images_per_id:
        existing = {row["image_path"] for row in selected}
        rest = sorted(
            [row for row in rows if row["image_path"] not in existing],
            key=quality_key,
        )
        selected.extend(rest[: max(0, images_per_id - len(selected))])

    return selected[:images_per_id]


def select_jobs(
    metadata: list[dict[str, Any]],
    vehicle_ids: list[str],
    limit_ids: int,
    images_per_id: int,
    require_both_views: bool,
) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped = group_by_vehicle(metadata)
    if vehicle_ids:
        candidate_ids = vehicle_ids
    else:
        candidate_ids = sorted(grouped)

    jobs: list[tuple[str, list[dict[str, Any]]]] = []
    for vehicle_id in candidate_ids:
        rows = grouped.get(vehicle_id, [])
        if not rows:
            continue
        views = {row.get("view_source") for row in rows}
        if require_both_views and not {"ground_camera", "uav"}.issubset(views):
            continue
        selected = select_representative_images(rows, images_per_id)
        if selected:
            jobs.append((vehicle_id, selected))
        if len(jobs) >= limit_ids:
            break

    if not jobs:
        raise SystemExit("No annotation jobs selected from metadata.")
    return jobs


def build_user_content(
    vehicle_id: str,
    rows: list[dict[str, Any]],
    dataset_root: Path,
    max_image_side: int,
    jpeg_quality: int,
    pipeline: str,
    caption_variants: int,
    semantic_match_threshold: float,
) -> list[dict[str, Any]]:
    image_meta = []
    for idx, row in enumerate(rows, start=1):
        image_meta.append(
            {
                "image_index": idx,
                "image_path": row["image_path"],
                "vehicle_id": row["vehicle_id"],
                "camera_id": row["camera_id"],
                "view_source": row["view_source"],
                "platform_type": row["platform_type"],
                "target_size": row["target_size"],
                "small_target": row["small_target"],
                "source_split": row.get("source_split"),
            }
        )

    alternate_count = max(0, caption_variants - 1)
    schema = {
        "perception": {
            "images": [
                {
                    "image_index": "integer matching the provided image_index",
                    "image_path": "exact provided image_path",
                    "attributes": [
                        {
                            "name": "one vehicle attribute name",
                            "value": "controlled value or uncertain",
                            "confidence": "number from 0 to 1",
                            "visibility": "visible|partial|not_visible|uncertain",
                            "visual_evidence": "brief visible evidence, not hidden reasoning",
                            "source_images": ["the current exact image_path"],
                        }
                    ],
                    "quality_notes": ["small target, blur, occlusion, crop, or distractor notes"],
                }
            ]
        },
        "cross_view_consensus": {
            "stable": [
                {
                    "attribute": "attribute name",
                    "value": "agreed value",
                    "confidence": "number from 0 to 1",
                    "source_images": ["supporting image_path values"],
                }
            ],
            "view_specific": ["attributes visible in only one view"],
            "conflict": ["cross-image or cross-view contradictions"],
            "uncertain": ["attributes without enough evidence"],
        },
        "annotation_id": {
            "dataset": "tag_vr",
            "vehicle_id": vehicle_id,
            "source_datasets": ["cvpair"],
            "identity_scope": "global",
            "identity_confidence": "global_id_verified",
            "description_zh": "60-100 个中文汉字、1-2 句、属性密集的 ID 级稳定描述",
            "description_en": "35-60 English words in 1-2 information-dense sentences for stable ID-level cues",
            "description_zh_variants": f"array of exactly {alternate_count} natural paraphrase(s) with no new facts",
            "description_en_variants": f"array of exactly {alternate_count} natural paraphrase(s) with no new facts",
            "color": "white|black|gray|silver|red|blue|yellow|green|brown|other|uncertain",
            "vehicle_type": "one string from: " + "|".join(CONTROLLED_ENUMS["vehicle_type"]),
            "body_profile": "short controlled phrase or uncertain",
            "roof_features": ["controlled visible roof feature or uncertain"],
            "window_features": ["controlled visible window feature or uncertain"],
            "cargo_or_rear_structure": "short controlled phrase or uncertain",
            "special_marks": ["visible special mark, or empty"],
            "stable_attributes": ["stable vehicle attributes only"],
            "uncertain_attributes": ["fields that are uncertain"],
            "qa_status": "one string: auto_labeled|manual_review",
        },
        "annotations_image": [
            {
                "image_index": "integer matching the provided image_index",
                "image_path": "exact provided image_path",
                "vehicle_id": vehicle_id,
                "source_dataset": "cvpair",
                "camera_id": "c0|c1",
                "view_source": "ground_camera|uav",
                "platform_type": "ground_camera|uav",
                "color": "same controlled color vocabulary",
                "vehicle_type": "one string from: " + "|".join(CONTROLLED_ENUMS["vehicle_type"]),
                "orientation": "one string from: " + "|".join(CONTROLLED_ENUMS["orientation"]),
                "visible_parts": "array containing only visible items from: " + "|".join(CONTROLLED_ENUMS["visible_parts"]),
                "occlusion": "one string from: " + "|".join(CONTROLLED_ENUMS["occlusion"]),
                "target_size": "copy target_size from input",
                "small_target": "copy small_target from input",
                "scene_context": "short visible context, not identity feature",
                "weather": "clear|rain|fog|snow|uncertain|null",
                "illumination": "daylight|night|low_light|uncertain|null",
                "background_distractors": {
                    "neighbor_vehicle_count": "integer or null",
                    "occluders": ["visible occluder labels"],
                    "complexity": "low|medium|high|uncertain",
                },
                "description_zh": "50-90 个中文汉字、1-2 句、属性密集的图像级可见描述",
                "description_en": "30-55 English words in 1-2 information-dense sentences for visible image-level cues",
                "description_zh_variants": f"array of exactly {alternate_count} natural paraphrase(s) with no new facts",
                "description_en_variants": f"array of exactly {alternate_count} natural paraphrase(s) with no new facts",
                "confidence": "number from 0 to 1",
                "uncertain_fields": ["field names"],
                "qa_status": "one string: auto_labeled|manual_review",
            }
        ],
        "semantic_audit": {
            "id": {
                "category": "match|contradictory|hallucinatory|vacuous",
                "match_score": "number from 0 to 1",
                "field_issues": ["incorrect, missing, unsupported, or empty fields"],
                "corrections": ["concise corrections applied"],
                "omitted_attributes": ["visible attributes omitted from captions"],
            },
            "images": [
                {
                    "image_path": "exact provided image_path",
                    "category": "match|contradictory|hallucinatory|vacuous",
                    "match_score": "number from 0 to 1",
                    "field_issues": ["issues"],
                    "corrections": ["corrections"],
                    "omitted_attributes": ["visible omitted attributes"],
                }
            ],
            "overall_status": "pass|manual_review",
        },
        "qa_notes": ["short notes about uncertainty or potential review needs"],
    }

    instruction = {
        "task": "TAG-VR CVPair annotation",
        "annotation_method_version": ANNOTATION_METHOD_VERSION,
        "prompt_version": PROMPT_VERSION,
        "pipeline": pipeline,
        "vehicle_id": vehicle_id,
        "image_metadata": image_meta,
        "output_schema": schema,
        "controlled_enums": CONTROLLED_ENUMS,
        "stable_vehicle_attributes": STABLE_VEHICLE_ATTRIBUTES,
        "semantic_qa_categories": SEMANTIC_QA_CATEGORIES,
        "semantic_match_threshold": semantic_match_threshold,
        "caption_variants_total": caption_variants,
        "caption_length_rules": CAPTION_LENGTH_RULES,
        "caption_content_rules": {
            "id_level": [
                "Describe only stable cross-view vehicle cues, never background identity shortcuts.",
                "Prioritize color, coarse vehicle type, body profile and proportions, windows or windshield, roof or rear structure, and visible special marks.",
                "State which cues remain stable across views and explain why any attribute is uncertain.",
                "Use 1-2 natural, information-dense sentences rather than a field list or generic template.",
            ],
            "image_level": [
                "Describe the main vehicle first, including color, current view or orientation, visible body parts, and occlusion.",
                "Mention neighboring vehicles or scene clutter only as visible context or interference, never as identity cues.",
                "Explain uncertain fields using visible reasons such as viewpoint, occlusion, lighting, low resolution, or cross-view inconsistency.",
                "Use 1-2 natural, information-dense sentences and keep Chinese and English semantically aligned.",
            ],
        },
        "important": [
            "Return strict JSON only.",
            "annotations_image length must equal the number of provided images.",
            "Use exact image_path values from image_metadata.",
            "If uncertain, set the relevant field to uncertain and add it to uncertain_fields.",
            "Never infer brand, exact model, or license plate.",
            "Do not use background as stable identity attribute.",
            "ID captions must be 60-100 Chinese characters and 35-60 English words.",
            "Image captions must be 50-90 Chinese characters and 30-55 English words.",
            "Every caption must contain 1-2 information-dense sentences and must not be generic filler.",
            "All scalar categorical fields must be a single string; never return the enum option list itself.",
            "Perform per-image perception before captioning and cite brief visible evidence for every asserted stable attribute.",
            "Use ground images as semantic anchors but require cross-view support before writing ID-level stable cues.",
            "Return the requested number of canonical/paraphrase captions; paraphrases must not add facts.",
            "If any semantic audit category is not match or its score is below the threshold, set overall_status to manual_review.",
        ],
    }

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": json.dumps(instruction, ensure_ascii=False, indent=2),
        }
    ]
    for idx, row in enumerate(rows, start=1):
        image_path = dataset_root / row["image_path"]
        content.append(
            {
                "type": "text",
                "text": f"Image {idx}: {row['image_path']}",
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_data_url(
                        image_path,
                        max_side=max_image_side,
                        jpeg_quality=jpeg_quality,
                    ),
                    "detail": "low",
                },
            }
        )
    return content


def build_audit_user_content(
    vehicle_id: str,
    rows: list[dict[str, Any]],
    dataset_root: Path,
    initial_payload: dict[str, Any],
    max_image_side: int,
    jpeg_quality: int,
    caption_variants: int,
    semantic_match_threshold: float,
) -> list[dict[str, Any]]:
    instruction = {
        "task": "TAG-VR independent visual audit and caption refinement",
        "annotation_method_version": ANNOTATION_METHOD_VERSION,
        "prompt_version": PROMPT_VERSION,
        "vehicle_id": vehicle_id,
        "quality_categories": SEMANTIC_QA_CATEGORIES,
        "semantic_match_threshold": semantic_match_threshold,
        "caption_variants_total": caption_variants,
        "caption_length_rules": CAPTION_LENGTH_RULES,
        "initial_payload": initial_payload,
        "required_output": {
            "refined_annotation": {
                "annotation_id": "complete corrected annotation_id object",
                "annotations_image": "complete corrected annotations_image array with exact image_path values",
            },
            "semantic_audit": {
                "id": {
                    "category": "match|contradictory|hallucinatory|vacuous",
                    "match_score": "number from 0 to 1",
                    "field_issues": ["issues found"],
                    "corrections": ["corrections applied"],
                    "omitted_attributes": ["visible attributes added during refinement"],
                },
                "images": [
                    {
                        "image_path": "exact image_path",
                        "category": "match|contradictory|hallucinatory|vacuous",
                        "match_score": "number from 0 to 1",
                        "field_issues": ["issues found"],
                        "corrections": ["corrections applied"],
                        "omitted_attributes": ["visible attributes added"],
                    }
                ],
                "overall_status": "pass|manual_review",
            },
        },
        "audit_rules": [
            "Re-check every attribute and caption claim against the images.",
            "Correct contradictory, hallucinated, vacuous, or omitted content before returning refined_annotation.",
            "Keep ID-level captions limited to cross-view stable cues supported by multiple images.",
            "Keep image-level captions specific to the current view and visible parts.",
            "Preserve exactly the requested canonical plus paraphrase caption count without adding facts.",
            "Treat captions below the configured length minimum as vacuous and expand them before returning.",
            "Do not reuse the same caption for ID-level, ground-view, and UAV-view records.",
            "Use overall_status=pass only when every category is match and every score meets the threshold.",
        ],
    }
    content: list[dict[str, Any]] = [
        {"type": "text", "text": json.dumps(instruction, ensure_ascii=False, indent=2)}
    ]
    for idx, row in enumerate(rows, start=1):
        image_path = dataset_root / row["image_path"]
        content.append({"type": "text", "text": f"Audit image {idx}: {row['image_path']}"})
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_data_url(
                        image_path,
                        max_side=max_image_side,
                        jpeg_quality=jpeg_quality,
                    ),
                    "detail": "low",
                },
            }
        )
    return content


def compact_caption_qa_failures(qa_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact_rows = []
    for row in qa_rows:
        if row.get("status") == "pass":
            continue
        compact_rows.append(
            {
                "level": row.get("level"),
                "variant": row.get("variant"),
                "vehicle_id": row.get("vehicle_id"),
                "image_path": row.get("image_path"),
                "description_zh_characters": row.get("description_zh_characters"),
                "description_en_words": row.get("description_en_words"),
                "description_zh_sentences": row.get("description_zh_sentences"),
                "description_en_sentences": row.get("description_en_sentences"),
                "issues": row.get("issues", []),
                "rules": row.get("rules", {}),
            }
        )
    return compact_rows


def build_caption_repair_user_content(
    vehicle_id: str,
    rows: list[dict[str, Any]],
    current_payload: dict[str, Any],
    caption_qa_rows: list[dict[str, Any]],
    caption_variants: int,
) -> list[dict[str, Any]]:
    image_meta = [
        {
            "image_index": idx,
            "image_path": row["image_path"],
            "view_source": row["view_source"],
            "camera_id": row["camera_id"],
            "target_size": row["target_size"],
            "small_target": row["small_target"],
        }
        for idx, row in enumerate(rows, start=1)
    ]
    instruction = {
        "task": "TAG-VR caption length repair",
        "annotation_method_version": ANNOTATION_METHOD_VERSION,
        "prompt_version": PROMPT_VERSION,
        "vehicle_id": vehicle_id,
        "image_metadata": image_meta,
        "caption_variants_total": caption_variants,
        "caption_length_rules": CAPTION_LENGTH_RULES,
        "caption_qa_failures": compact_caption_qa_failures(caption_qa_rows),
        "current_payload": current_payload,
        "required_output": {
            "refined_annotation": {
                "annotation_id": "complete corrected annotation_id object with all non-caption fields preserved",
                "annotations_image": "complete corrected annotations_image array with exact image_path values and all non-caption fields preserved",
            },
            "semantic_audit": "copy or update semantic_audit only if the repaired captions change support status",
            "qa_notes": ["caption repair notes, if any"],
        },
        "repair_rules": [
            "Repair every canonical caption and every natural paraphrase listed in caption_qa_failures.",
            "Preserve all non-caption fields unless they are internally inconsistent with existing evidence.",
            "Use only facts already present in perception, cross_view_consensus, annotations_image fields, semantic_audit, and qa_notes.",
            "For ID-level captions, expand with stable cross-view cues: color, coarse type, body profile, proportions, window or windshield cues, roof or rear structure, visible special marks, and uncertainty reasons.",
            "For image-level captions, expand with current view, orientation, visible parts, occlusion, target clarity, neighboring vehicles or background clutter as interference, and uncertainty reasons.",
            "Do not add brand, exact model, license plate, or unseen details.",
            "Do not use the same sentence for ID-level and image-level captions; the UAV and ground captions must reflect their own visible view.",
            "Keep each Chinese caption within its Han-character band and each English caption within its word band.",
            "Return strict JSON only.",
        ],
    }
    return [{"type": "text", "text": json.dumps(instruction, ensure_ascii=False, indent=2)}]


def append_images_to_content(
    content: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    dataset_root: Path,
    max_image_side: int,
    jpeg_quality: int,
    label: str,
) -> None:
    for index, row in enumerate(rows, start=1):
        content.append({"type": "text", "text": f"{label} {index}: {row['image_path']}"})
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_data_url(
                        dataset_root / row["image_path"],
                        max_side=max_image_side,
                        jpeg_quality=jpeg_quality,
                    ),
                    "detail": "low",
                },
            }
        )


def build_v4_perception_content(
    vehicle_id: str,
    rows: list[dict[str, Any]],
    dataset_root: Path,
    max_image_side: int,
    jpeg_quality: int,
    requested_observations: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    instruction = {
        "task": "TAG-VR v4 per-image independent structured perception",
        "vehicle_id": vehicle_id,
        "annotation_method_version": ANNOTATION_METHOD_VERSION,
        "prompt_version": PROMPT_VERSION,
        "requested_observations_from_caption_stage": requested_observations or [],
        "image_metadata": [
            {
                "image_index": index,
                "image_path": row["image_path"],
                "view_source": row["view_source"],
                "camera_id": row["camera_id"],
                "small_target": row["small_target"],
                "target_size": row["target_size"],
            }
            for index, row in enumerate(rows, start=1)
        ],
        "required_output": {
            "perception": {
                "images": [
                    {
                        "image_index": "exact input index",
                        "image_path": "exact input path",
                        "attributes": [
                            {
                                "attribute": "single attribute name; use visible_part once per part",
                                "value": "single controlled or concise value",
                                "confidence": "0 to 1",
                                "visibility": "visible|partial|not_visible|uncertain|not_applicable",
                                "visual_evidence": "brief current-image evidence",
                                "source_images": ["current image_path only"],
                            }
                        ],
                        "quality_notes": ["blur, scale, occlusion, crop, or distractor notes"],
                    }
                ]
            }
        },
        "rules": [
            "Judge every image independently; do not transfer details from another view.",
            "Do not generate captions or cross-view stable attributes in this stage.",
            "Return one image record for every input image and use exact image_path values.",
            "Use separate visible_part records for roof, hood, windshield, windows, lights, wheels, side body, or body boundary.",
            "Every attribute must contain all required fields, and source_images must contain only the current image.",
            "Do not infer brand, exact model, or license plate.",
            "Return strict JSON only.",
        ],
    }
    content = [{"type": "text", "text": json.dumps(instruction, ensure_ascii=False, indent=2)}]
    append_images_to_content(
        content,
        rows,
        dataset_root,
        max_image_side,
        jpeg_quality,
        "Perception image",
    )
    return content


def build_v4_consensus_content(
    vehicle_id: str,
    perception: dict[str, Any],
    program_draft: dict[str, Any],
) -> list[dict[str, Any]]:
    instruction = {
        "task": "TAG-VR v4 VLM A synonym and conflict adjudication",
        "vehicle_id": vehicle_id,
        "program_stable_rule": program_draft.get("stable_rule"),
        "perception": perception,
        "program_consensus_draft": program_draft,
        "required_output": {
            "normalization_suggestions": [
                {
                    "attribute": "attribute name",
                    "raw_value": "unknown or synonymous observed value",
                    "canonical_value": "canonical equivalent value",
                    "reason": "why these are synonyms",
                }
            ],
            "conflict_explanations": [
                {"attribute": "attribute name", "explanation": "why values conflict"}
            ],
            "downgrade_suggestions": [
                {"attribute": "attribute name", "reason": "why evidence should remain uncertain"}
            ],
            "stable_suggestions": [],
        },
        "rules": [
            "Do not generate captions.",
            "Do not decide stable; stable_suggestions must remain empty.",
            "Only normalize true synonyms, explain conflicts, or recommend downgrade to uncertain.",
            "The deterministic program has final veto authority.",
            "Return strict JSON only.",
        ],
    }
    return [{"type": "text", "text": json.dumps(instruction, ensure_ascii=False, indent=2)}]


def caption_output_schema(vehicle_id: str, caption_variants: int) -> dict[str, Any]:
    alternate_count = max(0, caption_variants - 1)
    claim_schema = {
        "claim": "atomic natural-language fact represented in both zh/en captions",
        "claim_type": "positive|uncertainty",
        "attribute": "fact attribute or uncertain field",
        "value": "exact normalized fact value; uncertain for uncertainty claims",
        "fact_ids": ["exactly one authorized fact_id for a positive claim; empty for uncertainty"],
        "reason": "required for uncertainty claims",
    }
    return {
        "annotation_id": {
            "vehicle_id": vehicle_id,
            "description_zh": "60-100 Han characters, 1-2 sentences",
            "description_en": "35-60 English words, 1-2 sentences",
            "description_zh_variants": f"exactly {alternate_count} paraphrase(s)",
            "description_en_variants": f"exactly {alternate_count} paraphrase(s)",
            "caption_claims": {
                variant: [claim_schema] for variant in caption_variant_names(caption_variants)
            },
            "color": "controlled value",
            "vehicle_type": "controlled value",
            "body_profile": "controlled value",
            "roof_features": [],
            "window_features": [],
            "cargo_or_rear_structure": "controlled value or uncertain",
            "special_marks": [],
            "stable_attributes": [],
            "uncertain_attributes": [],
            "qa_status": "auto_labeled|manual_review",
        },
        "annotations_image": [
            {
                "image_path": "exact input path",
                "vehicle_id": vehicle_id,
                "camera_id": "input camera_id",
                "view_source": "input view_source",
                "platform_type": "input platform_type",
                "description_zh": "50-90 Han characters, 1-2 sentences",
                "description_en": "30-55 English words, 1-2 sentences",
                "description_zh_variants": f"exactly {alternate_count} paraphrase(s)",
                "description_en_variants": f"exactly {alternate_count} paraphrase(s)",
                "caption_claims": {
                    variant: [claim_schema] for variant in caption_variant_names(caption_variants)
                },
                "color": "authorized fact value or uncertain",
                "vehicle_type": "authorized fact value or uncertain",
                "orientation": "authorized fact value or uncertain",
                "visible_parts": [],
                "occlusion": "authorized fact value or uncertain",
                "scene_context": "context only, never identity cue",
                "confidence": "0 to 1",
                "uncertain_fields": [],
                "qa_status": "auto_labeled|manual_review",
            }
        ],
        "new_observations": [
            {
                "image_path": "exact input path",
                "attribute": "attribute absent from fact table",
                "proposed_value": "observed value",
                "visual_evidence": "brief evidence",
            }
        ],
    }


def build_v4_caption_content(
    vehicle_id: str,
    rows: list[dict[str, Any]],
    dataset_root: Path,
    payload: dict[str, Any],
    max_image_side: int,
    jpeg_quality: int,
    caption_variants: int,
    id_coverage_threshold: float,
    image_coverage_threshold: float,
) -> list[dict[str, Any]]:
    fact_tables = build_authorized_fact_tables(payload, rows, payload.get("evidence_confidence_threshold", EVIDENCE_CONFIDENCE_THRESHOLD))
    instruction = {
        "task": "TAG-VR v4 evidence-locked bilingual caption generation",
        "vehicle_id": vehicle_id,
        "authoritative_fact_tables": fact_tables,
        "cross_view_consensus": payload.get("cross_view_consensus", {}),
        "image_metadata": [
            {
                "image_path": row["image_path"],
                "camera_id": row["camera_id"],
                "view_source": row["view_source"],
                "platform_type": row["platform_type"],
                "target_size": row["target_size"],
                "small_target": row["small_target"],
            }
            for row in rows
        ],
        "coverage_thresholds": {
            "id": id_coverage_threshold,
            "image": image_coverage_threshold,
            "uncertainty": UNCERTAINTY_COVERAGE_THRESHOLD,
        },
        "required_output": caption_output_schema(vehicle_id, caption_variants),
        "rules": [
            "The fact tables are authoritative; images may only verify phrasing.",
            "Use one atomic positive caption_claim per fact_id and exactly one authorized fact_id per positive claim.",
            "Any image observation absent from the fact table must go to new_observations and must not appear in captions.",
            "ID captions may cite only ID stable fact_ids; image captions may cite only fact_ids for that image.",
            "If at least three visible_part facts exist, cover at least three; otherwise cover all and add a visible_parts_limit uncertainty claim with a reason.",
            "Explain every listed uncertainty using an uncertainty claim with a visible reason.",
            "Do not use not_visible, not_applicable, conflict, or unsupported facts as positive claims.",
            "Canonical and paraphrase variants must have the same fact coverage and no new facts.",
            "Return strict JSON only.",
        ],
    }
    content = [{"type": "text", "text": json.dumps(instruction, ensure_ascii=False, indent=2)}]
    append_images_to_content(
        content,
        rows,
        dataset_root,
        max_image_side,
        jpeg_quality,
        "Caption verification image",
    )
    return content


def captions_for_independent_audit(payload: dict[str, Any]) -> dict[str, Any]:
    annotation_id = payload.get("annotation_id") if isinstance(payload.get("annotation_id"), dict) else {}
    raw_images = payload.get("annotations_image") if isinstance(payload.get("annotations_image"), list) else []
    return {
        "annotation_id": {
            key: annotation_id.get(key)
            for key in (
                "vehicle_id",
                "description_zh",
                "description_en",
                "description_zh_variants",
                "description_en_variants",
            )
        },
        "annotations_image": [
            {
                key: item.get(key)
                for key in (
                    "image_path",
                    "vehicle_id",
                    "view_source",
                    "description_zh",
                    "description_en",
                    "description_zh_variants",
                    "description_en_variants",
                )
            }
            for item in raw_images
            if isinstance(item, dict)
        ],
    }


def claim_audit_output_schema(caption_variants: int) -> dict[str, Any]:
    audit_item = {
        "variant": "canonical or natural_paraphrase_N",
        "all_caption_claims_covered": True,
        "claims": [
            {
                "claim": "one independently extracted caption claim",
                "attribute": "claim attribute",
                "value": "claimed value",
                "status": "supported|contradicted|not_visible",
                "source_images": ["supporting or checked image paths"],
                "correction": "required when status is not supported",
            }
        ],
    }
    return {
        "id": [audit_item for _ in caption_variant_names(caption_variants)],
        "images": [
            {"image_path": "exact image path", **audit_item}
            for _ in caption_variant_names(caption_variants)
        ],
    }


def build_v4_claim_audit_content(
    vehicle_id: str,
    rows: list[dict[str, Any]],
    dataset_root: Path,
    payload: dict[str, Any],
    local_qa: dict[str, Any],
    max_image_side: int,
    jpeg_quality: int,
    caption_variants: int,
    audit_round: int,
) -> list[dict[str, Any]]:
    fact_tables = build_authorized_fact_tables(
        payload,
        rows,
        payload.get("evidence_confidence_threshold", EVIDENCE_CONFIDENCE_THRESHOLD),
    )
    instruction = {
        "task": "TAG-VR v4 independent claim-level visual audit",
        "audit_round": audit_round,
        "vehicle_id": vehicle_id,
        "captions": captions_for_independent_audit(payload),
        "evidence_tables_without_confidence": redact_fact_table_confidence(fact_tables),
        "local_qa_status": local_qa.get("status"),
        "required_output": claim_audit_output_schema(caption_variants),
        "rules": [
            "Independently extract every claim from every Chinese/English canonical and paraphrase caption.",
            "Do not rely on VLM A caption_claims; they are intentionally hidden.",
            "Use supported only when visible evidence and the evidence table support the claim.",
            "Use contradicted for direct conflicts and not_visible when the claim cannot be verified in the cited images.",
            "Return source_images for every claim and a correction for every failed claim.",
            "Set all_caption_claims_covered=true only after auditing all claims in that variant.",
            "Return strict JSON only.",
        ],
    }
    content = [{"type": "text", "text": json.dumps(instruction, ensure_ascii=False, indent=2)}]
    append_images_to_content(
        content,
        rows,
        dataset_root,
        max_image_side,
        jpeg_quality,
        f"Audit round {audit_round} image",
    )
    return content


def compact_local_qa_failures(local_qa: dict[str, Any]) -> dict[str, Any]:
    return {
        key: [item for item in local_qa.get(key, []) if item.get("status") != "pass"]
        for key in ("schema_qa", "fact_coverage_qa", "caption_length_qa")
    }


def build_v4_audit_repair_content(
    vehicle_id: str,
    rows: list[dict[str, Any]],
    payload: dict[str, Any],
    local_qa: dict[str, Any],
    claim_audit: dict[str, Any],
    caption_variants: int,
) -> list[dict[str, Any]]:
    fact_tables = build_authorized_fact_tables(
        payload,
        rows,
        payload.get("evidence_confidence_threshold", EVIDENCE_CONFIDENCE_THRESHOLD),
    )
    instruction = {
        "task": "TAG-VR v4 VLM A caption repair after local QA and VLM B audit",
        "vehicle_id": vehicle_id,
        "authoritative_fact_tables": fact_tables,
        "current_annotation": {
            "annotation_id": payload.get("annotation_id", {}),
            "annotations_image": payload.get("annotations_image", []),
        },
        "local_qa_failures": compact_local_qa_failures(local_qa),
        "vlm_b_claim_failures": [
            item
            for item in claim_audit.get("claim_rows", [])
            if item.get("qa_status") != "pass"
        ],
        "vlm_b_summary_failures": [
            item
            for item in claim_audit.get("summary_rows", [])
            if item.get("status") != "pass"
        ],
        "required_output": {
            "annotation_id": caption_output_schema(vehicle_id, caption_variants)["annotation_id"],
            "annotations_image": caption_output_schema(vehicle_id, caption_variants)["annotations_image"],
            "new_observations": [],
        },
        "rules": [
            "Repair every listed local-QA or VLM-B issue.",
            "Use only authorized fact_ids and preserve atomic caption_claims.",
            "Do not modify perception, consensus, or fact tables.",
            "Do not add new observations during repair; unsupported content must be removed or replaced with supported facts.",
            "Return strict JSON only.",
        ],
    }
    return [{"type": "text", "text": json.dumps(instruction, ensure_ascii=False, indent=2)}]


def call_json_model_stage(
    stage: str,
    model: str,
    base_url: str,
    api_key: str,
    system_prompt: str,
    content: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "response_format": {"type": "json_object"},
    }
    response = post_chat_completion(
        base_url=base_url,
        api_key=api_key,
        payload=request_payload,
        timeout=args.timeout,
        retries=args.retries,
    )
    response_text = extract_message_text(response)
    parsed = parse_json_object(response_text)
    raw_stage = {
        "stage": stage,
        "model": model,
        "base_url": base_url,
        "system_prompt": system_prompt,
        "user_instruction": content[0]["text"],
        "response_text": response_text,
        "usage": response.get("usage"),
        "finish_reason": (response.get("choices") or [{}])[0].get("finish_reason"),
    }
    return parsed, raw_stage


def strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = strip_json_fence(text)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Model output is not a JSON object.")
    return payload


def post_chat_completion(
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code}: {error_body[:1000]}")
        except Exception as exc:  # noqa: BLE001 - preserve network diagnostics.
            last_error = exc
        if attempt < retries:
            time.sleep(2.0 * (attempt + 1))
    assert last_error is not None
    raise last_error


def extract_message_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise ValueError("API response has no choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        if parts:
            return "\n".join(parts)
    raise ValueError("API response message has no text content.")


def normalize_id_annotation(
    payload: dict[str, Any],
    vehicle_id: str,
    caption_variants: int = 2,
) -> dict[str, Any]:
    ann = payload.get("annotation_id") or payload.get("annotations_id") or {}
    if not isinstance(ann, dict):
        ann = {}
    normalized = {
        "dataset": "tag_vr",
        "vehicle_id": vehicle_id,
        "source_datasets": ["cvpair"],
        "identity_scope": "global",
        "identity_confidence": "global_id_verified",
        "description_zh": ensure_scalar(ann.get("description_zh"), default=""),
        "description_en": ensure_scalar(ann.get("description_en"), default=""),
        "color": ensure_scalar(ann.get("color")),
        "vehicle_type": ensure_scalar(ann.get("vehicle_type")),
        "body_profile": ensure_scalar(ann.get("body_profile")),
        "roof_features": ensure_list(ann.get("roof_features")),
        "window_features": ensure_list(ann.get("window_features")),
        "cargo_or_rear_structure": ensure_scalar(ann.get("cargo_or_rear_structure")),
        "special_marks": ensure_list(ann.get("special_marks")),
        "stable_attributes": ensure_list(ann.get("stable_attributes")),
        "uncertain_attributes": ensure_list(ann.get("uncertain_attributes")),
        "caption_claims": ann.get("caption_claims")
        if isinstance(ann.get("caption_claims"), dict)
        else {},
        "review_reasons": ensure_list(ann.get("review_reasons")),
        "qa_status": ensure_scalar(ann.get("qa_status"), default="auto_labeled"),
    }
    normalized["description_zh_variants"] = normalize_caption_variants(
        ann.get("description_zh_variants"),
        canonical=normalized["description_zh"],
        total_count=caption_variants,
    )
    normalized["description_en_variants"] = normalize_caption_variants(
        ann.get("description_en_variants"),
        canonical=normalized["description_en"],
        total_count=caption_variants,
    )
    if not normalized["description_zh"] or not normalized["description_en"]:
        normalized["qa_status"] = "manual_review"
        if "description" not in normalized["uncertain_attributes"]:
            normalized["uncertain_attributes"].append("description")
    return normalized


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return [
            item
            for item in value
            if item is not None and str(item).strip() and str(item).strip().lower() != "null"
        ]
    if isinstance(value, str) and value.strip().lower() == "null":
        return []
    if isinstance(value, str) and not value.strip():
        return []
    return [value]


def normalize_caption_variants(
    value: Any,
    canonical: str,
    total_count: int,
) -> list[str]:
    desired_alternates = max(0, total_count - 1)
    variants = []
    for item in ensure_list(value):
        text = ensure_scalar(item, default="")
        if not text or text == canonical or text in variants:
            continue
        variants.append(text)
    return variants[:desired_alternates]


def ensure_scalar(value: Any, default: str = "uncertain") -> str:
    if isinstance(value, list):
        for item in value:
            scalar = ensure_scalar(item, default="")
            if scalar:
                return scalar
        return default
    if value is None:
        return default
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() == "null":
            return default
        return stripped
    return str(value)


def normalize_optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() == "null":
            return None
        return stripped
    return str(value)


def normalize_confidence(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


ATTRIBUTE_NAME_ALIASES = {
    "colour": "color",
    "vehicle_colour": "color",
    "type": "vehicle_type",
    "coarse_type": "vehicle_type",
    "body_shape": "body_profile",
    "body_outline": "body_profile",
    "body_ratio": "body_proportions",
    "roof_features": "roof_feature",
    "window_features": "window_feature",
    "special_marks": "special_mark",
    "visible_parts": "visible_part",
    "parts": "visible_part",
}


ATTRIBUTE_VALUE_ALIASES = {
    "color": {
        "grey": "gray",
        "silver_gray": "silver",
        "silver_grey": "silver",
        "silvergray": "silver",
        "silvergrey": "silver",
    },
    "vehicle_type": {
        "car": "passenger_vehicle",
        "passenger_car": "passenger_vehicle",
        "compact_car": "compact_passenger_vehicle",
        "compact_passenger_car": "compact_passenger_vehicle",
        "minibus": "van_minibus",
        "van": "van_minibus",
        "lighttruck": "light_truck",
        "boxtruck": "box_truck",
    },
    "body_profile": {
        "compact_low": "compact_low",
        "low_compact": "compact_low",
        "low_profile_compact": "compact_low",
        "compact_and_low": "compact_low",
    },
    "visibility": {
        "partially_visible": "partial",
        "partly_visible": "partial",
        "invisible": "not_visible",
        "not_applicable": "not_applicable",
        "n_a": "not_applicable",
    },
}


KNOWN_CANONICAL_VALUES = {
    "color": set(COLOR_ZH) if "COLOR_ZH" in globals() else {
        "white", "black", "gray", "silver", "red", "blue", "yellow", "green", "brown", "other", "uncertain"
    },
    "vehicle_type": set(CONTROLLED_ENUMS["vehicle_type"]) | {
        "passenger_vehicle",
        "compact_passenger_vehicle",
    },
    "orientation": set(CONTROLLED_ENUMS["orientation"]),
    "visible_part": set(CONTROLLED_ENUMS["visible_parts"]),
    "occlusion": set(CONTROLLED_ENUMS["occlusion"]),
}


CROSS_VIEW_STABLE_ATTRIBUTES = {
    "color",
    "vehicle_type",
    "body_profile",
    "body_proportions",
    "roof_feature",
    "window_feature",
    "cargo_or_rear_structure",
    "special_mark",
}


def normalize_label_token(value: Any) -> str:
    text = ensure_scalar(value, default="uncertain").strip().lower()
    text = re.sub(r"[^a-z0-9\u3400-\u4dbf\u4e00-\u9fff]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_") or "uncertain"


def normalize_attribute_name(value: Any) -> str:
    token = normalize_label_token(value)
    return ATTRIBUTE_NAME_ALIASES.get(token, token)


def normalize_attribute_value(attribute: str, value: Any) -> str:
    token = normalize_label_token(value)
    return ATTRIBUTE_VALUE_ALIASES.get(attribute, {}).get(token, token)


def normalize_visibility(value: Any) -> str:
    token = normalize_attribute_value("visibility", value)
    if token in POSITIVE_VISIBILITY_VALUES | EXCLUDED_VISIBILITY_VALUES | {"uncertain"}:
        return token
    return "uncertain"


def is_ground_view(view_source: Any) -> bool:
    value = ensure_scalar(view_source, default="unknown").lower()
    return value == "ground_camera" or value.startswith("ground") or value.startswith("road_side")


def is_uav_view(view_source: Any) -> bool:
    value = ensure_scalar(view_source, default="unknown").lower()
    return value == "uav" or value.startswith("aerial")


def normalize_perception_payload(
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    raw_perception = payload.get("perception") if isinstance(payload.get("perception"), dict) else payload
    raw_images = raw_perception.get("images") if isinstance(raw_perception, dict) else []
    if not isinstance(raw_images, list):
        raw_images = []
    raw_by_path = {
        item.get("image_path"): item
        for item in raw_images
        if isinstance(item, dict) and isinstance(item.get("image_path"), str)
    }
    raw_by_index = {
        item.get("image_index"): item
        for item in raw_images
        if isinstance(item, dict) and isinstance(item.get("image_index"), int)
    }
    normalized_images = []
    schema_issues = []
    evidence_counter = 0
    for image_index, row in enumerate(rows, start=1):
        image_path = row["image_path"]
        raw_image = raw_by_path.get(image_path) or raw_by_index.get(image_index) or {}
        raw_attributes = raw_image.get("attributes") if isinstance(raw_image, dict) else []
        if not isinstance(raw_image, dict) or "quality_notes" not in raw_image:
            schema_issues.append(f"missing_quality_notes:{image_path}")
        if not isinstance(raw_attributes, list):
            raw_attributes = []
        if not raw_attributes:
            schema_issues.append(f"missing_attributes:{image_path}")
        normalized_attributes = []
        for raw_attribute in raw_attributes:
            if not isinstance(raw_attribute, dict):
                schema_issues.append(f"invalid_attribute:{image_path}")
                continue
            required_fields = {
                "attribute",
                "value",
                "confidence",
                "visibility",
                "visual_evidence",
                "source_images",
            }
            missing_fields = sorted(required_fields - set(raw_attribute))
            if missing_fields:
                schema_issues.append(
                    f"missing_perception_fields:{image_path}:{','.join(missing_fields)}"
                )
            attribute = normalize_attribute_name(
                raw_attribute.get("attribute", raw_attribute.get("name"))
            )
            raw_value = raw_attribute.get("value", "uncertain")
            values = raw_value if isinstance(raw_value, list) else [raw_value]
            if attribute == "visible_part" and not values:
                values = ["uncertain"]
            for value in values:
                evidence_counter += 1
                confidence = normalize_confidence(raw_attribute.get("confidence"))
                visibility = normalize_visibility(raw_attribute.get("visibility"))
                visual_evidence = ensure_scalar(raw_attribute.get("visual_evidence"), default="")
                if not visual_evidence:
                    schema_issues.append(
                        f"missing_visual_evidence:{image_path}:{attribute}:{evidence_counter}"
                    )
                provided_sources = ensure_list(raw_attribute.get("source_images"))
                if provided_sources != [image_path]:
                    schema_issues.append(
                        f"source_images_rewritten_to_current_image:{image_path}:{attribute}:{evidence_counter}"
                    )
                normalized_attributes.append(
                    {
                        "evidence_id": f"e{image_index:02d}_{evidence_counter:04d}",
                        "attribute": attribute,
                        "raw_value": ensure_scalar(value),
                        "value": normalize_attribute_value(attribute, value),
                        "confidence": confidence,
                        "visibility": visibility,
                        "visual_evidence": visual_evidence,
                        "source_images": [image_path],
                    }
                )
        normalized_images.append(
            {
                "image_index": image_index,
                "image_path": image_path,
                "view_source": row.get("view_source"),
                "attributes": normalized_attributes,
                "quality_notes": ensure_list(raw_image.get("quality_notes"))
                if isinstance(raw_image, dict)
                else [],
            }
        )
    return {"images": normalized_images, "schema_issues": list(dict.fromkeys(schema_issues))}


def eligible_positive_evidence(record: dict[str, Any], confidence_threshold: float) -> bool:
    return (
        normalize_confidence(record.get("confidence")) >= confidence_threshold
        and record.get("visibility") in POSITIVE_VISIBILITY_VALUES
        and record.get("value") not in {"", "uncertain", "not_visible", "not_applicable"}
    )


def build_program_consensus_draft(
    perception: dict[str, Any],
    confidence_threshold: float = EVIDENCE_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for image in perception.get("images", []):
        if not isinstance(image, dict):
            continue
        for attribute in image.get("attributes", []):
            if not isinstance(attribute, dict):
                continue
            record = dict(attribute)
            record["image_path"] = image.get("image_path")
            record["view_source"] = image.get("view_source")
            grouped[ensure_scalar(record.get("attribute"))].append(record)

    candidates = []
    for attribute, observations in sorted(grouped.items()):
        eligible = [
            record
            for record in observations
            if eligible_positive_evidence(record, confidence_threshold)
        ]
        by_value: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in eligible:
            by_value[ensure_scalar(record.get("value"))].append(record)
        values = sorted(by_value)
        ground_paths = sorted(
            {
                ensure_scalar(record.get("image_path"), default="")
                for record in eligible
                if is_ground_view(record.get("view_source"))
            }
            - {""}
        )
        uav_paths = sorted(
            {
                ensure_scalar(record.get("image_path"), default="")
                for record in eligible
                if is_uav_view(record.get("view_source"))
            }
            - {""}
        )
        value_consistent = len(values) == 1
        unresolved_conflict = len(values) > 1
        eligible_for_stable = (
            attribute in CROSS_VIEW_STABLE_ATTRIBUTES
            and bool(ground_paths)
            and bool(uav_paths)
            and value_consistent
            and not unresolved_conflict
        )
        if attribute not in CROSS_VIEW_STABLE_ATTRIBUTES and eligible:
            candidate_status = "view_specific"
        elif eligible_for_stable:
            candidate_status = "stable_candidate"
        elif unresolved_conflict:
            candidate_status = "conflict"
        elif eligible:
            candidate_status = "view_specific"
        else:
            candidate_status = "uncertain"
        candidates.append(
            {
                "attribute": attribute,
                "candidate_status": candidate_status,
                "values": values,
                "ground_support_count": len(ground_paths),
                "uav_support_count": len(uav_paths),
                "ground_source_images": ground_paths,
                "uav_source_images": uav_paths,
                "value_consistent": value_consistent,
                "unresolved_conflict": unresolved_conflict,
                "program_stable_eligible": eligible_for_stable,
                "observations": [
                    {
                        "evidence_id": record.get("evidence_id"),
                        "image_path": record.get("image_path"),
                        "view_source": record.get("view_source"),
                        "raw_value": record.get("raw_value"),
                        "value": record.get("value"),
                        "confidence": record.get("confidence"),
                        "visibility": record.get("visibility"),
                    }
                    for record in observations
                ],
            }
        )
    return {
        "rule_owner": "deterministic_program",
        "confidence_threshold": confidence_threshold,
        "stable_rule": (
            "ground_support_count>=1 AND uav_support_count>=1 AND "
            "normalized_value_consistent AND no_unresolved_conflict"
        ),
        "candidates": candidates,
    }


def apply_vlm_normalization_suggestions(
    perception: dict[str, Any],
    adjudication: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    updated = copy.deepcopy(perception)
    accepted = []
    rejected = []
    suggestions = adjudication.get("normalization_suggestions")
    if not isinstance(suggestions, list):
        suggestions = []
    for suggestion in suggestions:
        if not isinstance(suggestion, dict):
            continue
        attribute = normalize_attribute_name(suggestion.get("attribute"))
        raw_value = normalize_attribute_value(attribute, suggestion.get("raw_value"))
        canonical_value = normalize_attribute_value(attribute, suggestion.get("canonical_value"))
        changed = False
        reject_reason = None
        if canonical_value in {"", "uncertain", "not_visible", "not_applicable"}:
            reject_reason = "invalid_canonical_value"
        elif raw_value in KNOWN_CANONICAL_VALUES.get(attribute, set()) and raw_value != canonical_value:
            reject_reason = "cannot_remap_known_canonical_value"
        else:
            for image in updated.get("images", []):
                for record in image.get("attributes", []):
                    if record.get("attribute") != attribute:
                        continue
                    if record.get("value") != raw_value and normalize_attribute_value(
                        attribute, record.get("raw_value")
                    ) != raw_value:
                        continue
                    record["value"] = canonical_value
                    record["normalization_source"] = "vlm_a_synonym_suggestion"
                    changed = True
        audit_record = {
            "attribute": attribute,
            "raw_value": raw_value,
            "canonical_value": canonical_value,
            "reason": ensure_scalar(suggestion.get("reason"), default=""),
        }
        if changed and not reject_reason:
            accepted.append(audit_record)
        else:
            audit_record["rejection_reason"] = reject_reason or "no_matching_observation"
            rejected.append(audit_record)
    return updated, accepted, rejected


def finalize_cross_view_consensus(
    perception: dict[str, Any],
    adjudication: dict[str, Any],
    confidence_threshold: float = EVIDENCE_CONFIDENCE_THRESHOLD,
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized_perception, accepted, rejected = apply_vlm_normalization_suggestions(
        perception,
        adjudication,
    )
    draft = build_program_consensus_draft(normalized_perception, confidence_threshold)
    downgrade_attributes = {
        normalize_attribute_name(item.get("attribute") if isinstance(item, dict) else item)
        for item in ensure_list(adjudication.get("downgrade_suggestions"))
    }
    conflict_explanations = {
        normalize_attribute_name(item.get("attribute")): ensure_scalar(
            item.get("explanation"), default="unresolved cross-view conflict"
        )
        for item in ensure_list(adjudication.get("conflict_explanations"))
        if isinstance(item, dict)
    }
    stable = []
    view_specific = []
    conflict = []
    uncertain = []
    for candidate in draft["candidates"]:
        attribute = candidate["attribute"]
        observations = candidate["observations"]
        source_images = sorted(
            {
                ensure_scalar(item.get("image_path"), default="")
                for item in observations
            }
            - {""}
        )
        evidence_ids = sorted(
            {
                ensure_scalar(item.get("evidence_id"), default="")
                for item in observations
                if item.get("evidence_id")
            }
        )
        if candidate["program_stable_eligible"] and attribute not in downgrade_attributes:
            stable.append(
                {
                    "fact_id": f"id:{attribute}:{candidate['values'][0]}",
                    "attribute": attribute,
                    "value": candidate["values"][0],
                    "source_images": source_images,
                    "evidence_ids": evidence_ids,
                    "ground_support_count": candidate["ground_support_count"],
                    "uav_support_count": candidate["uav_support_count"],
                    "rule_decision": "stable",
                }
            )
        elif attribute not in CROSS_VIEW_STABLE_ATTRIBUTES:
            view_specific.append(
                {
                    "attribute": attribute,
                    "values": candidate["values"],
                    "source_images": source_images,
                    "evidence_ids": evidence_ids,
                    "ground_support_count": candidate["ground_support_count"],
                    "uav_support_count": candidate["uav_support_count"],
                    "rule_decision": "view_specific",
                }
            )
        elif candidate["unresolved_conflict"]:
            conflict.append(
                {
                    "attribute": attribute,
                    "values": candidate["values"],
                    "source_images": source_images,
                    "evidence_ids": evidence_ids,
                    "explanation": conflict_explanations.get(
                        attribute,
                        "normalized values remain inconsistent",
                    ),
                    "rule_decision": "conflict",
                }
            )
        elif attribute in downgrade_attributes or candidate["candidate_status"] == "uncertain":
            uncertain.append(
                {
                    "attribute": attribute,
                    "values": candidate["values"],
                    "source_images": source_images,
                    "evidence_ids": evidence_ids,
                    "reason": "VLM A suggested downgrade" if attribute in downgrade_attributes else "insufficient eligible evidence",
                    "rule_decision": "uncertain",
                }
            )
        else:
            view_specific.append(
                {
                    "attribute": attribute,
                    "values": candidate["values"],
                    "source_images": source_images,
                    "evidence_ids": evidence_ids,
                    "ground_support_count": candidate["ground_support_count"],
                    "uav_support_count": candidate["uav_support_count"],
                    "rule_decision": "view_specific",
                }
            )
    ignored_upgrade_suggestions = ensure_list(adjudication.get("stable_suggestions"))
    consensus = {
        "rule_owner": "deterministic_program",
        "stable_rule": draft["stable_rule"],
        "stable": stable,
        "view_specific": view_specific,
        "conflict": conflict,
        "uncertain": uncertain,
        "vlm_adjudication": {
            "accepted_normalizations": accepted,
            "rejected_normalizations": rejected,
            "ignored_upgrade_suggestions": ignored_upgrade_suggestions,
            "downgrade_attributes": sorted(downgrade_attributes),
        },
    }
    return consensus, normalized_perception


def build_authorized_fact_tables(
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
    confidence_threshold: float = EVIDENCE_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    perception = payload.get("perception") if isinstance(payload.get("perception"), dict) else {}
    consensus = (
        payload.get("cross_view_consensus")
        if isinstance(payload.get("cross_view_consensus"), dict)
        else {}
    )
    evidence_lookup = {}
    for image in perception.get("images", []):
        if not isinstance(image, dict):
            continue
        for item in image.get("attributes", []):
            if isinstance(item, dict) and item.get("evidence_id"):
                evidence_lookup[item["evidence_id"]] = item
    id_facts = []
    for item in consensus.get("stable", []):
        if not isinstance(item, dict):
            continue
        attribute = normalize_attribute_name(item.get("attribute"))
        value = normalize_attribute_value(attribute, item.get("value"))
        fact_id = ensure_scalar(item.get("fact_id"), default=f"id:{attribute}:{value}")
        evidence_ids = ensure_list(item.get("evidence_ids"))
        id_facts.append(
            {
                "fact_id": fact_id,
                "attribute": attribute,
                "value": value,
                "source_images": ensure_list(item.get("source_images")),
                "evidence_ids": evidence_ids,
                "evidence": [
                    {
                        "evidence_id": evidence_id,
                        "attribute": evidence_lookup.get(evidence_id, {}).get("attribute"),
                        "value": evidence_lookup.get(evidence_id, {}).get("value"),
                        "confidence": evidence_lookup.get(evidence_id, {}).get("confidence"),
                        "visibility": evidence_lookup.get(evidence_id, {}).get("visibility"),
                        "visual_evidence": evidence_lookup.get(evidence_id, {}).get(
                            "visual_evidence"
                        ),
                        "source_images": evidence_lookup.get(evidence_id, {}).get(
                            "source_images", []
                        ),
                    }
                    for evidence_id in evidence_ids
                    if evidence_id in evidence_lookup
                ],
                "applicability": "applicable",
            }
        )

    id_uncertainties = []
    for group in ("uncertain", "conflict"):
        for item in consensus.get(group, []):
            if not isinstance(item, dict):
                continue
            attribute = normalize_attribute_name(item.get("attribute"))
            id_uncertainties.append(
                {
                    "uncertainty_id": f"id_uncertain:{attribute}",
                    "attribute": attribute,
                    "reason": ensure_scalar(
                        item.get("reason", item.get("explanation")),
                        default="cross-view evidence is insufficient or conflicting",
                    ),
                    "source_images": ensure_list(item.get("source_images")),
                    "group": group,
                }
            )

    image_facts: dict[str, list[dict[str, Any]]] = {row["image_path"]: [] for row in rows}
    image_uncertainties: dict[str, list[dict[str, Any]]] = {
        row["image_path"]: [] for row in rows
    }
    for image in perception.get("images", []):
        if not isinstance(image, dict):
            continue
        image_path = ensure_scalar(image.get("image_path"), default="")
        if image_path not in image_facts:
            continue
        for item in image.get("attributes", []):
            if not isinstance(item, dict):
                continue
            attribute = normalize_attribute_name(item.get("attribute"))
            value = normalize_attribute_value(attribute, item.get("value"))
            visibility = normalize_visibility(item.get("visibility"))
            confidence = normalize_confidence(item.get("confidence"))
            evidence_id = ensure_scalar(item.get("evidence_id"), default="")
            if eligible_positive_evidence(item, confidence_threshold):
                image_facts[image_path].append(
                    {
                        "fact_id": evidence_id,
                        "attribute": attribute,
                        "value": value,
                        "confidence": confidence,
                        "visibility": visibility,
                        "visual_evidence": ensure_scalar(
                            item.get("visual_evidence"), default=""
                        ),
                        "source_images": [image_path],
                        "applicability": "applicable",
                    }
                )
            elif visibility not in EXCLUDED_VISIBILITY_VALUES and (
                visibility == "uncertain"
                or value == "uncertain"
                or confidence < confidence_threshold
            ):
                image_uncertainties[image_path].append(
                    {
                        "uncertainty_id": f"{image_path}:uncertain:{attribute}:{evidence_id}",
                        "attribute": attribute,
                        "reason": ensure_scalar(
                            item.get("visual_evidence"),
                            default="visible evidence is insufficient",
                        ),
                        "source_images": [image_path],
                    }
                )

    def dedupe(items: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
        seen = set()
        result = []
        for item in items:
            value = item.get(key)
            if not value or value in seen:
                continue
            seen.add(value)
            result.append(item)
        return result

    return {
        "confidence_threshold": confidence_threshold,
        "id": {
            "facts": dedupe(id_facts, "fact_id"),
            "uncertainties": dedupe(id_uncertainties, "uncertainty_id"),
        },
        "images": {
            image_path: {
                "facts": dedupe(facts, "fact_id"),
                "uncertainties": dedupe(
                    image_uncertainties.get(image_path, []),
                    "uncertainty_id",
                ),
            }
            for image_path, facts in image_facts.items()
        },
    }


def redact_fact_table_confidence(fact_tables: dict[str, Any]) -> dict[str, Any]:
    def redact(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: redact(item)
                for key, item in value.items()
                if key not in {"confidence", "confidence_threshold"}
            }
        if isinstance(value, list):
            return [redact(item) for item in value]
        return value

    return redact(fact_tables)


def caption_variant_names(caption_variants: int) -> list[str]:
    return ["canonical"] + [
        f"natural_paraphrase_{index}"
        for index in range(1, max(0, caption_variants - 1) + 1)
    ]


def annotation_claims_for_variant(annotation: dict[str, Any], variant: str) -> list[dict[str, Any]]:
    raw = annotation.get("caption_claims")
    if not isinstance(raw, dict):
        return []
    claims = raw.get(variant)
    if not isinstance(claims, list):
        return []
    return [item for item in claims if isinstance(item, dict)]


def annotation_text_for_variant(
    annotation: dict[str, Any],
    language: str,
    variant: str,
) -> str:
    if variant == "canonical":
        return ensure_scalar(annotation.get(f"description_{language}"), default="")
    variants = annotation.get(f"description_{language}_variants")
    if not isinstance(variants, list):
        return ""
    try:
        index = int(variant.rsplit("_", 1)[1]) - 1
    except (IndexError, ValueError):
        return ""
    return ensure_scalar(variants[index], default="") if index < len(variants) else ""


def build_v4_schema_qa_rows(
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
    vehicle_id: str,
    caption_variants: int,
) -> list[dict[str, Any]]:
    qa_rows = []
    annotation_id = payload.get("annotation_id") if isinstance(payload.get("annotation_id"), dict) else {}
    raw_images = payload.get("annotations_image") if isinstance(payload.get("annotations_image"), list) else []
    image_by_path = {
        item.get("image_path"): item
        for item in raw_images
        if isinstance(item, dict) and isinstance(item.get("image_path"), str)
    }

    def validate(level: str, image_path: Optional[str], annotation: dict[str, Any]) -> None:
        if level == "id":
            required_annotation_fields = {
                "description_zh",
                "description_en",
                "description_zh_variants",
                "description_en_variants",
                "caption_claims",
                "color",
                "vehicle_type",
                "body_profile",
                "stable_attributes",
                "uncertain_attributes",
                "qa_status",
            }
        else:
            required_annotation_fields = {
                "image_path",
                "vehicle_id",
                "camera_id",
                "view_source",
                "platform_type",
                "description_zh",
                "description_en",
                "description_zh_variants",
                "description_en_variants",
                "caption_claims",
                "color",
                "vehicle_type",
                "orientation",
                "visible_parts",
                "occlusion",
                "scene_context",
                "confidence",
                "uncertain_fields",
                "qa_status",
            }
        record_issues = [
            f"missing_annotation_field:{field}"
            for field in sorted(required_annotation_fields - set(annotation))
        ]
        for variant in caption_variant_names(caption_variants):
            issues = list(record_issues)
            if not annotation_text_for_variant(annotation, "zh", variant):
                issues.append("missing_description_zh")
            if not annotation_text_for_variant(annotation, "en", variant):
                issues.append("missing_description_en")
            claims = annotation_claims_for_variant(annotation, variant)
            if not claims:
                issues.append("missing_caption_claims")
            for index, claim in enumerate(claims):
                claim_type = ensure_scalar(claim.get("claim_type"), default="")
                if claim_type not in {"positive", "uncertainty"}:
                    issues.append(f"invalid_claim_type:{index}")
                if not ensure_scalar(claim.get("claim"), default=""):
                    issues.append(f"missing_claim_text:{index}")
                if not ensure_scalar(claim.get("attribute"), default=""):
                    issues.append(f"missing_claim_attribute:{index}")
                if claim_type == "positive" and len(ensure_list(claim.get("fact_ids"))) != 1:
                    issues.append(f"positive_claim_must_reference_one_fact:{index}")
                if claim_type == "uncertainty" and not ensure_scalar(
                    claim.get("reason"), default=""
                ):
                    issues.append(f"missing_uncertainty_reason:{index}")
            qa_rows.append(
                {
                    "level": level,
                    "vehicle_id": vehicle_id,
                    "image_path": image_path,
                    "variant": variant,
                    "issues": list(dict.fromkeys(issues)),
                    "status": "pass" if not issues else "manual_review",
                }
            )

    validate("id", None, annotation_id)
    for row in rows:
        validate("image", row["image_path"], image_by_path.get(row["image_path"], {}))

    perception = payload.get("perception") if isinstance(payload.get("perception"), dict) else {}
    consensus = payload.get("cross_view_consensus") if isinstance(payload.get("cross_view_consensus"), dict) else {}
    stage_issues = ensure_list(perception.get("schema_issues"))
    if consensus.get("rule_owner") != "deterministic_program":
        stage_issues.append("consensus_rule_owner_not_program")
    if payload.get("new_observations"):
        stage_issues.append("unresolved_new_observations")
    qa_rows.append(
        {
            "level": "pipeline",
            "vehicle_id": vehicle_id,
            "image_path": None,
            "variant": "pipeline",
            "issues": list(dict.fromkeys(stage_issues)),
            "status": "pass" if not stage_issues else "manual_review",
        }
    )
    return qa_rows


def build_fact_coverage_qa_rows(
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
    vehicle_id: str,
    caption_variants: int,
    evidence_confidence_threshold: float = EVIDENCE_CONFIDENCE_THRESHOLD,
    id_coverage_threshold: float = ID_COVERAGE_THRESHOLD,
    image_coverage_threshold: float = IMAGE_COVERAGE_THRESHOLD,
    uncertainty_coverage_threshold: float = UNCERTAINTY_COVERAGE_THRESHOLD,
) -> list[dict[str, Any]]:
    fact_tables = build_authorized_fact_tables(
        payload,
        rows,
        confidence_threshold=evidence_confidence_threshold,
    )
    annotation_id = payload.get("annotation_id") if isinstance(payload.get("annotation_id"), dict) else {}
    raw_images = payload.get("annotations_image") if isinstance(payload.get("annotations_image"), list) else []
    image_by_path = {
        item.get("image_path"): item
        for item in raw_images
        if isinstance(item, dict) and isinstance(item.get("image_path"), str)
    }

    def evaluate(
        level: str,
        image_path: Optional[str],
        annotation: dict[str, Any],
        table: dict[str, Any],
        coverage_threshold: float,
    ) -> list[dict[str, Any]]:
        expected_facts = table.get("facts") if isinstance(table.get("facts"), list) else []
        expected_uncertainties = (
            table.get("uncertainties") if isinstance(table.get("uncertainties"), list) else []
        )
        fact_by_id = {
            fact.get("fact_id"): fact
            for fact in expected_facts
            if isinstance(fact, dict) and fact.get("fact_id")
        }
        uncertainty_attributes = {
            normalize_attribute_name(item.get("attribute"))
            for item in expected_uncertainties
            if isinstance(item, dict)
        }
        expected_part_ids = {
            fact_id
            for fact_id, fact in fact_by_id.items()
            if fact.get("attribute") == "visible_part"
        }
        result = []
        for variant in caption_variant_names(caption_variants):
            claims = annotation_claims_for_variant(annotation, variant)
            covered_fact_ids = set()
            explained_uncertainties = set()
            unsupported_claims = []
            for claim in claims:
                claim_type = ensure_scalar(claim.get("claim_type"), default="")
                attribute = normalize_attribute_name(claim.get("attribute"))
                value = normalize_attribute_value(attribute, claim.get("value"))
                fact_ids = ensure_list(claim.get("fact_ids"))
                if claim_type == "positive":
                    if len(fact_ids) != 1:
                        unsupported_claims.append(
                            {"claim": claim.get("claim"), "reason": "positive_claim_not_atomic"}
                        )
                        continue
                    fact = fact_by_id.get(fact_ids[0])
                    if fact is None:
                        unsupported_claims.append(
                            {"claim": claim.get("claim"), "reason": "unknown_fact_id", "fact_ids": fact_ids}
                        )
                        continue
                    if attribute != fact.get("attribute") or value != fact.get("value"):
                        unsupported_claims.append(
                            {
                                "claim": claim.get("claim"),
                                "reason": "claim_fact_mismatch",
                                "fact_ids": fact_ids,
                            }
                        )
                        continue
                    covered_fact_ids.add(fact_ids[0])
                elif claim_type == "uncertainty":
                    reason = ensure_scalar(claim.get("reason"), default="")
                    if attribute in uncertainty_attributes and reason:
                        explained_uncertainties.add(attribute)
                    elif attribute not in {"visible_parts_limit", "visible_part_limit"}:
                        unsupported_claims.append(
                            {"claim": claim.get("claim"), "reason": "unauthorized_uncertainty_claim"}
                        )
                else:
                    unsupported_claims.append(
                        {"claim": claim.get("claim"), "reason": "invalid_claim_type"}
                    )

            coverage_score = (
                len(covered_fact_ids) / len(fact_by_id) if fact_by_id else 1.0
            )
            uncertainty_coverage = (
                len(explained_uncertainties) / len(uncertainty_attributes)
                if uncertainty_attributes
                else 1.0
            )
            covered_part_ids = covered_fact_ids & expected_part_ids
            visible_part_limit_reason = any(
                ensure_scalar(claim.get("claim_type"), default="") == "uncertainty"
                and normalize_attribute_name(claim.get("attribute"))
                in {"visible_parts_limit", "visible_part_limit"}
                and bool(ensure_scalar(claim.get("reason"), default=""))
                for claim in claims
            )
            if level == "id":
                visible_parts_pass = True
            elif len(expected_part_ids) >= 3:
                visible_parts_pass = len(covered_part_ids) >= 3
            else:
                visible_parts_pass = (
                    covered_part_ids == expected_part_ids and visible_part_limit_reason
                )
            issues = []
            if coverage_score < coverage_threshold:
                issues.append("fact_coverage_below_threshold")
            if uncertainty_coverage < uncertainty_coverage_threshold:
                issues.append("uncertainty_coverage_below_threshold")
            if unsupported_claims:
                issues.append("unsupported_claim")
            if not visible_parts_pass:
                issues.append("visible_parts_coverage")
            result.append(
                {
                    "level": level,
                    "vehicle_id": vehicle_id,
                    "image_path": image_path,
                    "variant": variant,
                    "expected_fact_ids": sorted(fact_by_id),
                    "covered_fact_ids": sorted(covered_fact_ids),
                    "coverage_score": round(coverage_score, 6),
                    "coverage_threshold": coverage_threshold,
                    "expected_uncertainty_attributes": sorted(uncertainty_attributes),
                    "explained_uncertainty_attributes": sorted(explained_uncertainties),
                    "uncertainty_coverage": round(uncertainty_coverage, 6),
                    "uncertainty_coverage_threshold": uncertainty_coverage_threshold,
                    "expected_visible_part_fact_ids": sorted(expected_part_ids),
                    "covered_visible_part_fact_ids": sorted(covered_part_ids),
                    "visible_parts_pass": visible_parts_pass,
                    "unsupported_claims": unsupported_claims,
                    "issues": issues,
                    "status": "pass" if not issues else "manual_review",
                }
            )
        return result

    qa_rows = evaluate(
        "id",
        None,
        annotation_id,
        fact_tables["id"],
        id_coverage_threshold,
    )
    for row in rows:
        image_path = row["image_path"]
        qa_rows.extend(
            evaluate(
                "image",
                image_path,
                image_by_path.get(image_path, {}),
                fact_tables["images"].get(image_path, {"facts": [], "uncertainties": []}),
                image_coverage_threshold,
            )
        )
    return qa_rows


def run_v4_local_qa(
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
    vehicle_id: str,
    caption_variants: int,
    evidence_confidence_threshold: float,
    id_coverage_threshold: float,
    image_coverage_threshold: float,
    uncertainty_coverage_threshold: float,
) -> dict[str, Any]:
    schema_rows = build_v4_schema_qa_rows(payload, rows, vehicle_id, caption_variants)
    coverage_rows = build_fact_coverage_qa_rows(
        payload,
        rows,
        vehicle_id,
        caption_variants,
        evidence_confidence_threshold=evidence_confidence_threshold,
        id_coverage_threshold=id_coverage_threshold,
        image_coverage_threshold=image_coverage_threshold,
        uncertainty_coverage_threshold=uncertainty_coverage_threshold,
    )
    temp_id = normalize_id_annotation(payload, vehicle_id, caption_variants=caption_variants)
    temp_images = normalize_image_annotations(
        payload,
        rows,
        vehicle_id,
        caption_variants=caption_variants,
    )
    length_rows = apply_caption_length_qa(
        [temp_id],
        temp_images,
        expected_caption_variants=caption_variants,
    )
    all_pass = (
        all(row["status"] == "pass" for row in schema_rows)
        and all(row["status"] == "pass" for row in coverage_rows)
        and all(row["status"] == "pass" for row in length_rows)
    )
    return {
        "schema_qa": schema_rows,
        "fact_coverage_qa": coverage_rows,
        "caption_length_qa": length_rows,
        "status": "pass" if all_pass else "manual_review",
    }


def normalize_claim_audit(
    audit_payload: dict[str, Any],
    rows: list[dict[str, Any]],
    vehicle_id: str,
    caption_variants: int,
    audit_round: int,
) -> dict[str, Any]:
    raw_id = audit_payload.get("id")
    raw_id_items = raw_id if isinstance(raw_id, list) else []
    raw_images = audit_payload.get("images")
    raw_image_items = raw_images if isinstance(raw_images, list) else []
    id_by_variant = {
        ensure_scalar(item.get("variant"), default="canonical"): item
        for item in raw_id_items
        if isinstance(item, dict)
    }
    image_by_key = {
        (
            ensure_scalar(item.get("image_path"), default=""),
            ensure_scalar(item.get("variant"), default="canonical"),
        ): item
        for item in raw_image_items
        if isinstance(item, dict)
    }
    claim_rows = []
    summary_rows = []

    def normalize_item(
        level: str,
        image_path: Optional[str],
        variant: str,
        item: Any,
    ) -> None:
        issues = []
        claims = item.get("claims") if isinstance(item, dict) else []
        if not isinstance(claims, list) or not claims:
            claims = []
            issues.append("missing_claim_audit")
        if not isinstance(item, dict) or item.get("all_caption_claims_covered") is not True:
            issues.append("audit_did_not_confirm_full_claim_coverage")
        supported_count = 0
        for claim_index, claim in enumerate(claims):
            if not isinstance(claim, dict):
                issues.append(f"invalid_claim_record:{claim_index}")
                continue
            status = ensure_scalar(claim.get("status"), default="not_visible").lower()
            if status not in {"supported", "contradicted", "not_visible"}:
                status = "not_visible"
                issues.append(f"invalid_claim_status:{claim_index}")
            if status == "supported":
                supported_count += 1
            else:
                issues.append(f"claim_{status}:{claim_index}")
            claim_rows.append(
                {
                    "audit_round": audit_round,
                    "level": level,
                    "vehicle_id": vehicle_id,
                    "image_path": image_path,
                    "variant": variant,
                    "claim_index": claim_index,
                    "claim": ensure_scalar(claim.get("claim"), default=""),
                    "attribute": normalize_attribute_name(claim.get("attribute")),
                    "value": normalize_attribute_value(
                        normalize_attribute_name(claim.get("attribute")),
                        claim.get("value"),
                    ),
                    "status": status,
                    "source_images": ensure_list(claim.get("source_images")),
                    "correction": ensure_scalar(claim.get("correction"), default=""),
                    "qa_status": "pass" if status == "supported" else "manual_review",
                }
            )
        issues = list(dict.fromkeys(issues))
        summary_rows.append(
            {
                "audit_round": audit_round,
                "level": level,
                "vehicle_id": vehicle_id,
                "image_path": image_path,
                "variant": variant,
                "claim_count": len(claims),
                "supported_claim_count": supported_count,
                "issues": issues,
                "status": "pass" if not issues else "manual_review",
            }
        )

    for variant in caption_variant_names(caption_variants):
        normalize_item("id", None, variant, id_by_variant.get(variant))
    for row in rows:
        image_path = row["image_path"]
        for variant in caption_variant_names(caption_variants):
            normalize_item(
                "image",
                image_path,
                variant,
                image_by_key.get((image_path, variant)),
            )
    all_pass = all(item["status"] == "pass" for item in summary_rows)
    return {
        "audit_round": audit_round,
        "claim_rows": claim_rows,
        "summary_rows": summary_rows,
        "status": "pass" if all_pass else "manual_review",
        "raw": audit_payload,
    }


def semantic_audit_from_claim_audit(
    normalized_audit: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summary_rows = normalized_audit.get("summary_rows", [])

    def aggregate(level: str, image_path: Optional[str]) -> dict[str, Any]:
        relevant_summaries = [
            item
            for item in summary_rows
            if item.get("level") == level and item.get("image_path") == image_path
        ]
        relevant_claims = [
            item
            for item in normalized_audit.get("claim_rows", [])
            if item.get("level") == level and item.get("image_path") == image_path
        ]
        statuses = [item.get("status") for item in relevant_claims]
        if relevant_summaries and all(item.get("status") == "pass" for item in relevant_summaries):
            category = "match"
            score = 1.0
        elif "contradicted" in statuses:
            category = "contradictory"
            score = 0.0
        elif "not_visible" in statuses:
            category = "hallucinatory"
            score = 0.0
        else:
            category = "vacuous"
            score = 0.0
        corrections = [
            item.get("correction")
            for item in relevant_claims
            if item.get("correction")
        ]
        issues = [issue for item in relevant_summaries for issue in item.get("issues", [])]
        return {
            "category": category,
            "match_score": score,
            "field_issues": list(dict.fromkeys(issues)),
            "corrections": list(dict.fromkeys(corrections)),
            "omitted_attributes": [],
        }

    return {
        "id": aggregate("id", None),
        "images": [
            {"image_path": row["image_path"], **aggregate("image", row["image_path"])}
            for row in rows
        ],
        "overall_status": "pass"
        if normalized_audit.get("status") == "pass"
        else "manual_review",
    }


def select_final_claim_audit_summaries(
    summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    final_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in summary_rows:
        key = (
            row.get("vehicle_id"),
            row.get("level"),
            row.get("image_path"),
            row.get("variant"),
        )
        existing = final_by_key.get(key)
        if existing is None or int(row.get("audit_round") or 0) >= int(
            existing.get("audit_round") or 0
        ):
            final_by_key[key] = row
    return list(final_by_key.values())


def count_chinese_characters(text: str) -> int:
    return len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", text))


def count_english_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", text))


def count_sentences(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return len([part for part in re.split(r"[.!?。！？]+", stripped) if part.strip()])


def build_caption_length_qa_record(
    level: str,
    row: dict[str, Any],
    variant: str = "canonical",
    description_zh: Optional[str] = None,
    description_en: Optional[str] = None,
) -> dict[str, Any]:
    rules = CAPTION_LENGTH_RULES[level]
    if description_zh is None:
        description_zh = ensure_scalar(row.get("description_zh"), default="")
    if description_en is None:
        description_en = ensure_scalar(row.get("description_en"), default="")
    zh_count = count_chinese_characters(description_zh)
    en_count = count_english_words(description_en)
    zh_sentences = count_sentences(description_zh)
    en_sentences = count_sentences(description_en)
    issues = []
    if not rules["zh_min"] <= zh_count <= rules["zh_max"]:
        issues.append("description_zh_length")
    if not rules["en_min"] <= en_count <= rules["en_max"]:
        issues.append("description_en_length")
    if not rules["sentence_min"] <= zh_sentences <= rules["sentence_max"]:
        issues.append("description_zh_sentence_count")
    if not rules["sentence_min"] <= en_sentences <= rules["sentence_max"]:
        issues.append("description_en_sentence_count")

    return {
        "level": level,
        "variant": variant,
        "vehicle_id": row.get("vehicle_id"),
        "image_path": row.get("image_path"),
        "description_zh_characters": zh_count,
        "description_en_words": en_count,
        "description_zh_sentences": zh_sentences,
        "description_en_sentences": en_sentences,
        "rules": rules,
        "issues": issues,
        "status": "pass" if not issues else "manual_review",
    }


def apply_caption_length_qa(
    id_rows: list[dict[str, Any]],
    image_rows: list[dict[str, Any]],
    expected_caption_variants: int = 2,
) -> list[dict[str, Any]]:
    qa_rows = []
    for level, rows in (("id", id_rows), ("image", image_rows)):
        for row in rows:
            row_qa = [build_caption_length_qa_record(level, row)]
            zh_variants = [ensure_scalar(value, default="") for value in row.get("description_zh_variants", [])]
            en_variants = [ensure_scalar(value, default="") for value in row.get("description_en_variants", [])]
            for index in range(max(0, expected_caption_variants - 1)):
                description_zh = zh_variants[index] if index < len(zh_variants) else ""
                description_en = en_variants[index] if index < len(en_variants) else ""
                qa_row = build_caption_length_qa_record(
                    level,
                    row,
                    variant=f"natural_paraphrase_{index + 1}",
                    description_zh=description_zh,
                    description_en=description_en,
                )
                if not description_zh:
                    qa_row["issues"].append("missing_description_zh_variant")
                if not description_en:
                    qa_row["issues"].append("missing_description_en_variant")
                qa_row["issues"] = list(dict.fromkeys(qa_row["issues"]))
                qa_row["status"] = "pass" if not qa_row["issues"] else "manual_review"
                row_qa.append(qa_row)

            if any(qa_row["status"] != "pass" for qa_row in row_qa):
                row["qa_status"] = "manual_review"
                reasons = ensure_list(row.get("review_reasons"))
                if "caption_length_or_variant_qa" not in reasons:
                    reasons.append("caption_length_or_variant_qa")
                row["review_reasons"] = reasons
            for qa_row in row_qa:
                qa_row["annotation_qa_status"] = row.get("qa_status")
                qa_rows.append(qa_row)
    return qa_rows


def build_semantic_qa_rows(
    payload: dict[str, Any],
    vehicle_id: str,
    image_rows: list[dict[str, Any]],
    match_threshold: float,
) -> list[dict[str, Any]]:
    audit = payload.get("semantic_audit")
    if not isinstance(audit, dict):
        audit = {}
    id_audit = audit.get("id") if isinstance(audit.get("id"), dict) else {}
    raw_image_audits = audit.get("images") if isinstance(audit.get("images"), list) else []
    image_audits = {
        item.get("image_path"): item
        for item in raw_image_audits
        if isinstance(item, dict) and isinstance(item.get("image_path"), str)
    }

    def normalize_audit(level: str, image_path: Optional[str], item: dict[str, Any]) -> dict[str, Any]:
        category = ensure_scalar(item.get("category"), default="not_run").lower()
        score = normalize_confidence(item.get("match_score"))
        status = "pass" if category == "match" and score >= match_threshold else "manual_review"
        return {
            "level": level,
            "vehicle_id": vehicle_id,
            "image_path": image_path,
            "category": category,
            "match_score": score,
            "match_threshold": match_threshold,
            "field_issues": ensure_list(item.get("field_issues")),
            "corrections": ensure_list(item.get("corrections")),
            "omitted_attributes": ensure_list(item.get("omitted_attributes")),
            "status": status,
        }

    qa_rows = [normalize_audit("id", None, id_audit)]
    for image_row in image_rows:
        image_path = image_row["image_path"]
        qa_rows.append(
            normalize_audit("image", image_path, image_audits.get(image_path, {}))
        )
    return qa_rows


def apply_semantic_qa(
    id_row: dict[str, Any],
    image_rows: list[dict[str, Any]],
    semantic_qa_rows: list[dict[str, Any]],
) -> None:
    image_by_path = {row["image_path"]: row for row in image_rows}
    for qa_row in semantic_qa_rows:
        if qa_row["level"] == "id":
            target = id_row
        else:
            target = image_by_path.get(qa_row.get("image_path"))
        if target is None:
            continue
        target["semantic_qa_category"] = qa_row["category"]
        target["semantic_match_score"] = qa_row["match_score"]
        if qa_row["status"] != "pass":
            target["qa_status"] = "manual_review"
            reasons = ensure_list(target.get("review_reasons"))
            if "semantic_qa" not in reasons:
                reasons.append("semantic_qa")
            target["review_reasons"] = reasons


def apply_v4_quality_qa(
    id_row: dict[str, Any],
    image_rows: list[dict[str, Any]],
    schema_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    claim_audit: dict[str, Any],
) -> None:
    image_by_path = {row["image_path"]: row for row in image_rows}

    def target_for(level: str, image_path: Optional[str]) -> Optional[dict[str, Any]]:
        return id_row if level == "id" else image_by_path.get(image_path or "")

    pipeline_schema_failed = any(
        row.get("level") == "pipeline" and row.get("status") != "pass"
        for row in schema_rows
    )
    for target in [id_row, *image_rows]:
        relevant_schema = [
            row
            for row in schema_rows
            if row.get("level") in {"id", "image"}
            and (
                (row.get("level") == "id" and target is id_row)
                or row.get("image_path") == target.get("image_path")
            )
        ]
        schema_pass = not pipeline_schema_failed and all(
            row.get("status") == "pass" for row in relevant_schema
        )
        target["schema_qa_status"] = "pass" if schema_pass else "manual_review"
        if not schema_pass:
            target["qa_status"] = "manual_review"
            reasons = ensure_list(target.get("review_reasons"))
            if "schema_qa" not in reasons:
                reasons.append("schema_qa")
            target["review_reasons"] = reasons

    for level in ("id", "image"):
        paths = [None] if level == "id" else list(image_by_path)
        for image_path in paths:
            target = target_for(level, image_path)
            if target is None:
                continue
            relevant = [
                row
                for row in coverage_rows
                if row.get("level") == level and row.get("image_path") == image_path
            ]
            coverage_pass = bool(relevant) and all(
                row.get("status") == "pass" for row in relevant
            )
            target["fact_coverage_qa_status"] = (
                "pass" if coverage_pass else "manual_review"
            )
            target["fact_coverage_score"] = min(
                (float(row.get("coverage_score") or 0.0) for row in relevant),
                default=0.0,
            )
            target["uncertainty_coverage"] = min(
                (float(row.get("uncertainty_coverage") or 0.0) for row in relevant),
                default=0.0,
            )
            target["unsupported_claim_count"] = sum(
                len(row.get("unsupported_claims", [])) for row in relevant
            )
            if not coverage_pass:
                target["qa_status"] = "manual_review"
                reasons = ensure_list(target.get("review_reasons"))
                if "fact_coverage_qa" not in reasons:
                    reasons.append("fact_coverage_qa")
                target["review_reasons"] = reasons

    for summary in claim_audit.get("summary_rows", []):
        target = target_for(summary.get("level"), summary.get("image_path"))
        if target is None:
            continue
        current = target.get("claim_audit_status", "pass")
        if summary.get("status") != "pass":
            current = "manual_review"
            target["qa_status"] = "manual_review"
            reasons = ensure_list(target.get("review_reasons"))
            if "vlm_b_claim_audit" not in reasons:
                reasons.append("vlm_b_claim_audit")
            target["review_reasons"] = reasons
        target["claim_audit_status"] = current


def is_evaluation_sample(row: dict[str, Any]) -> bool:
    if row.get("source_split") in {"query", "gallery"}:
        return True
    memberships = row.get("protocol_memberships") or []
    return any(
        isinstance(item, dict) and item.get("split") in {"query", "gallery"}
        for item in memberships
    )


def deterministic_review_selected(identifier: str, rate: float, seed: int) -> bool:
    if rate <= 0:
        return False
    if rate >= 1:
        return True
    digest = hashlib.sha256(f"{seed}:{identifier}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return value < rate


def apply_human_review_policy(
    id_row: dict[str, Any],
    image_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    enforce_eval_review: bool,
    train_review_rate: float = 0.0,
    review_sampling_seed: int = 20260629,
) -> None:
    evaluation_identity = any(is_evaluation_sample(row) for row in source_rows)
    train_review_selected = (
        not evaluation_identity
        and deterministic_review_selected(
            ensure_scalar(id_row.get("vehicle_id"), default="unknown"),
            train_review_rate,
            review_sampling_seed,
        )
    )
    id_row["review_required"] = bool(
        (enforce_eval_review and evaluation_identity) or train_review_selected
    )
    id_reasons = ensure_list(id_row.get("review_reasons"))
    if id_row["review_required"]:
        id_reason = "evaluation_identity" if evaluation_identity else "sampled_train_identity"
        if id_reason not in id_reasons:
            id_reasons.append(id_reason)
    id_row["review_reasons"] = id_reasons
    if id_row["review_required"]:
        id_row["qa_status"] = "manual_review"

    source_by_path = {row["image_path"]: row for row in source_rows}
    for image_row in image_rows:
        source = source_by_path.get(image_row["image_path"], {})
        review_required = bool(
            (enforce_eval_review and is_evaluation_sample(source))
            or train_review_selected
        )
        image_row["review_required"] = review_required
        image_reasons = ensure_list(image_row.get("review_reasons"))
        if review_required:
            reason = "evaluation_sample" if is_evaluation_sample(source) else "sampled_train_identity"
            if reason not in image_reasons:
                image_reasons.append(reason)
        image_row["review_reasons"] = image_reasons
        if review_required:
            image_row["qa_status"] = "manual_review"


def build_annotation_evidence_row(
    payload: dict[str, Any],
    vehicle_id: str,
    source_rows: list[dict[str, Any]],
    backend: str,
    pipeline: str,
) -> dict[str, Any]:
    perception = payload.get("perception")
    if not isinstance(perception, dict):
        perception = {}
    raw_perception_images = perception.get("images")
    if not isinstance(raw_perception_images, list):
        raw_perception_images = []
    expected_paths = {row["image_path"] for row in source_rows}
    observed_paths = {
        item.get("image_path")
        for item in raw_perception_images
        if isinstance(item, dict) and isinstance(item.get("image_path"), str)
    }
    evidence_issues: list[str] = []
    if observed_paths != expected_paths:
        evidence_issues.append("perception_image_coverage")

    required_attribute_fields = {
        "value",
        "confidence",
        "visibility",
        "visual_evidence",
        "source_images",
    }
    for image_item in raw_perception_images:
        if not isinstance(image_item, dict):
            evidence_issues.append("invalid_perception_image_record")
            continue
        image_path = ensure_scalar(image_item.get("image_path"), default="unknown")
        attributes = image_item.get("attributes")
        if not isinstance(attributes, list) or not attributes:
            evidence_issues.append(f"missing_attributes:{image_path}")
            continue
        for attribute in attributes:
            if not isinstance(attribute, dict):
                evidence_issues.append(f"invalid_attribute_record:{image_path}")
                continue
            missing = sorted(required_attribute_fields - set(attribute))
            if "attribute" not in attribute and "name" not in attribute:
                missing.append("attribute")
            if missing:
                name = ensure_scalar(
                    attribute.get("attribute", attribute.get("name")),
                    default="unknown",
                )
                evidence_issues.append(
                    f"missing_attribute_fields:{image_path}:{name}:{','.join(missing)}"
                )

    consensus = payload.get("cross_view_consensus")
    if not isinstance(consensus, dict):
        consensus = {}
    for group in ("stable", "view_specific", "conflict", "uncertain"):
        if not isinstance(consensus.get(group), list):
            evidence_issues.append(f"missing_consensus_group:{group}")

    source_by_path = {row["image_path"]: row for row in source_rows}
    for stable_item in consensus.get("stable", []) if isinstance(consensus.get("stable"), list) else []:
        if not isinstance(stable_item, dict):
            evidence_issues.append("invalid_stable_consensus_record")
            continue
        attribute = ensure_scalar(stable_item.get("attribute"), default="unknown")
        supporting_paths = set(ensure_list(stable_item.get("source_images")))
        supporting_views = {
            source_by_path[path].get("view_source")
            for path in supporting_paths
            if path in source_by_path
        }
        if not {"ground_camera", "uav"}.issubset(supporting_views):
            evidence_issues.append(f"stable_without_cross_view_support:{attribute}")

    evidence_issues = list(dict.fromkeys(evidence_issues))
    return {
        "dataset": "tag_vr",
        "source_dataset": "cvpair",
        "vehicle_id": vehicle_id,
        "annotation_method_version": ANNOTATION_METHOD_VERSION,
        "prompt_version": PROMPT_VERSION,
        "backend": backend,
        "pipeline": pipeline,
        "selected_images": [row["image_path"] for row in source_rows],
        "perception": perception,
        "cross_view_consensus": consensus,
        "fact_tables": payload.get("fact_tables", {}),
        "local_qa": payload.get("local_qa", {}),
        "claim_audit": payload.get("claim_audit", {}),
        "semantic_audit": payload.get("semantic_audit", {}),
        "qa_notes": ensure_list(payload.get("qa_notes")),
        "evidence_qa": {
            "status": "pass" if not evidence_issues else "manual_review",
            "issues": evidence_issues,
            "expected_image_count": len(expected_paths),
            "observed_image_count": len(observed_paths),
        },
    }


def apply_evidence_qa(
    id_row: dict[str, Any],
    image_rows: list[dict[str, Any]],
    evidence_row: dict[str, Any],
) -> None:
    evidence_qa = evidence_row.get("evidence_qa") or {}
    if evidence_qa.get("status") == "pass":
        return
    for row in [id_row, *image_rows]:
        row["qa_status"] = "manual_review"
        reasons = ensure_list(row.get("review_reasons"))
        if "evidence_qa" not in reasons:
            reasons.append("evidence_qa")
        row["review_reasons"] = reasons


COLOR_ZH = {
    "white": "白色",
    "black": "黑色",
    "gray": "灰色",
    "silver": "银灰色",
    "red": "红色",
    "blue": "蓝色",
    "yellow": "黄色",
    "green": "绿色",
    "brown": "棕色",
    "other": "其他颜色",
    "uncertain": "颜色不确定",
}


COLOR_EN = {
    "white": "white",
    "black": "black",
    "gray": "gray",
    "silver": "silver-gray",
    "red": "red",
    "blue": "blue",
    "yellow": "yellow",
    "green": "green",
    "brown": "brown",
    "other": "other-colored",
    "uncertain": "uncertain-color",
}


VEHICLE_TYPE_ZH = {
    "sedan": "轿车",
    "suv": "SUV",
    "hatchback": "两厢车",
    "van_minibus": "厢式车或小巴",
    "pickup": "皮卡",
    "light_truck": "轻型卡车",
    "box_truck": "厢式货车",
    "bus": "巴士",
    "special_vehicle": "特殊车辆",
    "compact passenger vehicle": "紧凑型乘用车",
    "uncertain": "车辆",
}


VEHICLE_TYPE_EN = {
    "sedan": "sedan",
    "suv": "SUV",
    "hatchback": "hatchback",
    "van_minibus": "van or minibus",
    "pickup": "pickup",
    "light_truck": "light truck",
    "box_truck": "box truck",
    "bus": "bus",
    "special_vehicle": "special vehicle",
    "compact passenger vehicle": "compact passenger vehicle",
    "uncertain": "vehicle",
}


VISIBLE_PART_ZH = {
    "roof": "车顶",
    "hood": "前机盖",
    "trunk": "后备厢区域",
    "windshield": "挡风玻璃",
    "rear_window": "后窗",
    "side_windows": "侧窗",
    "front_lights": "前灯",
    "rear_lights": "尾灯",
    "wheels": "车轮",
    "side_body": "车身侧面",
    "body_boundary": "车身边界",
}


VISIBLE_PART_EN = {
    "roof": "roof",
    "hood": "hood",
    "trunk": "trunk area",
    "windshield": "windshield",
    "rear_window": "rear window",
    "side_windows": "side windows",
    "front_lights": "front lights",
    "rear_lights": "rear lights",
    "wheels": "wheels",
    "side_body": "side body",
    "body_boundary": "body boundary",
}


def caption_color_labels(value: Any) -> tuple[str, str]:
    raw = ensure_scalar(value, default="uncertain").lower().replace("_", "-")
    if "silver" in raw:
        key = "silver"
    elif "gray" in raw or "grey" in raw:
        key = "gray"
    elif "white" in raw:
        key = "white"
    elif "black" in raw:
        key = "black"
    elif "red" in raw:
        key = "red"
    elif "blue" in raw:
        key = "blue"
    elif "yellow" in raw:
        key = "yellow"
    elif "green" in raw:
        key = "green"
    elif "brown" in raw:
        key = "brown"
    else:
        key = "uncertain"
    return COLOR_ZH.get(key, "颜色不确定"), COLOR_EN.get(key, "uncertain-color")


def caption_vehicle_type_labels(value: Any) -> tuple[str, str]:
    raw = ensure_scalar(value, default="uncertain").lower().replace("-", "_")
    if "compact passenger" in raw:
        key = "compact passenger vehicle"
    elif "sedan" in raw:
        key = "sedan"
    elif "suv" in raw:
        key = "suv"
    elif "hatchback" in raw:
        key = "hatchback"
    elif "van" in raw or "minibus" in raw:
        key = "van_minibus"
    elif "pickup" in raw:
        key = "pickup"
    elif "box" in raw and "truck" in raw:
        key = "box_truck"
    elif "truck" in raw:
        key = "light_truck"
    elif "bus" in raw:
        key = "bus"
    elif "special" in raw:
        key = "special_vehicle"
    else:
        key = "uncertain"
    return VEHICLE_TYPE_ZH[key], VEHICLE_TYPE_EN[key]


def caption_body_profile_labels(value: Any) -> tuple[str, str]:
    raw = ensure_scalar(value, default="uncertain").lower().replace("-", "_")
    if "compact" in raw and ("low" in raw or "lower" in raw):
        return "紧凑且较低", "compact and low"
    if "compact" in raw:
        return "紧凑", "compact"
    if "low" in raw or "lower" in raw:
        return "较低", "relatively low"
    if "box" in raw:
        return "偏方正", "boxy"
    if raw == "uncertain":
        return "轮廓仍需复核", "uncertain"
    return "较清晰", "clear"


def caption_occlusion_label(value: Any) -> tuple[str, str]:
    raw = ensure_scalar(value, default="uncertain").lower()
    if raw in {"none", "none_obvious", "no", "clear"}:
        return "无明显遮挡", "no obvious occlusion"
    if "partial" in raw or "neighbor" in raw:
        return "存在邻车或局部遮挡", "partial occlusion or neighboring-vehicle interference"
    if "heavy" in raw:
        return "遮挡较重", "heavy occlusion"
    return "遮挡情况仍需复核", "occlusion remains uncertain"


def caption_visible_parts(parts: Any, view_source: str) -> tuple[str, str]:
    values = ensure_list(parts) or visible_parts_for_view(view_source)
    zh_parts = []
    en_parts = []
    for item in values[:4]:
        key = ensure_scalar(item, default="").lower()
        zh_parts.append(VISIBLE_PART_ZH.get(key, key or "可见部件"))
        en_parts.append(VISIBLE_PART_EN.get(key, key.replace("_", " ") or "visible parts"))
    return "、".join(zh_parts), ", ".join(en_parts)


def make_dense_id_captions(annotation: dict[str, Any]) -> tuple[str, str, str, str]:
    color_zh, color_en = caption_color_labels(annotation.get("color"))
    type_zh, type_en = caption_vehicle_type_labels(annotation.get("vehicle_type"))
    profile_zh, profile_en = caption_body_profile_labels(annotation.get("body_profile"))
    zh = (
        f"一辆{color_zh}{type_zh}，车身轮廓{profile_zh}，挡风玻璃、车窗和车顶区域是主要可见线索；"
        f"跨视角稳定信息为{color_zh}车身、{type_zh}外观和{profile_zh}比例，细分车型、局部结构或特殊标记仍需人工复核。"
    )
    en = (
        f"A {color_en} {type_en} with a {profile_en} body profile, visible windshield, window areas, "
        f"and roof or upper-body cues. The stable cross-view cues are the {color_en} body, "
        f"{type_en} appearance, and {profile_en} proportions, while fine type, local structure, "
        "or special marks still require review."
    )
    zh_alt = (
        f"该车呈{color_zh}{type_zh}外观，整体车身{profile_zh}，多视角中可确认车身颜色、轮廓比例以及部分车窗或车顶线索；"
        "具体车型级别、局部结构和特殊标记因视角差异仍保留不确定性。"
    )
    en_alt = (
        f"This target appears as a {color_en} {type_en} with {profile_en} overall proportions and visible "
        "window or upper-body cues. Across views, the reliable evidence is the body color, coarse type, "
        "and outline, while fine structural details and special marks remain uncertain."
    )
    return zh, en, zh_alt, en_alt


def make_dense_image_captions(annotation: dict[str, Any]) -> tuple[str, str, str, str]:
    view_source = ensure_scalar(annotation.get("view_source"), default="ground_camera")
    color_zh, color_en = caption_color_labels(annotation.get("color"))
    type_zh, type_en = caption_vehicle_type_labels(annotation.get("vehicle_type"))
    profile_zh, profile_en = caption_body_profile_labels(annotation.get("body_profile"))
    parts_zh, parts_en = caption_visible_parts(annotation.get("visible_parts"), view_source)
    occ_zh, occ_en = caption_occlusion_label(annotation.get("occlusion"))
    if view_source == "uav":
        view_zh = "空中俯视图"
        view_en = "aerial top-view crop"
    else:
        view_zh = "地面近景"
        view_en = "ground-view crop"
    zh = (
        f"当前{view_zh}中主车为{color_zh}{type_zh}，可见{parts_zh}和{profile_zh}车身轮廓，{occ_zh}；"
        "邻近车辆或停车环境只作为背景干扰，细分车型、局部结构和不可见部件仍需复核。"
    )
    en = (
        f"In this {view_en}, the main vehicle is a {color_en} {type_en} with visible {parts_en} "
        f"and a {profile_en} body outline. The record shows {occ_en}; neighboring vehicles or "
        "parking context are only background interference, and fine type or hidden parts still need review."
    )
    zh_alt = (
        f"该{view_zh}显示一辆{color_zh}{type_zh}，主要可见{parts_zh}，车身比例{profile_zh}且{occ_zh}；"
        "背景中的邻车和停车场线索不作为身份特征，无法确认的细节保持不确定。"
    )
    en_alt = (
        f"From the {view_en}, the target appears as a {color_en} {type_en} with {parts_en} visible, "
        f"{profile_en} proportions, and {occ_en}. Nearby cars and parking-scene cues are distractors, "
        "so unresolved fine details should remain uncertain."
    )
    return zh, en, zh_alt, en_alt


def apply_local_caption_density_repair(
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
    vehicle_id: str,
    caption_variants: int,
) -> dict[str, Any]:
    repaired = dict(payload)
    annotation_id = dict(repaired.get("annotation_id") or {})
    id_zh, id_en, id_zh_alt, id_en_alt = make_dense_id_captions(annotation_id)
    annotation_id["description_zh"] = id_zh
    annotation_id["description_en"] = id_en
    annotation_id["description_zh_variants"] = [id_zh_alt] if caption_variants > 1 else []
    annotation_id["description_en_variants"] = [id_en_alt] if caption_variants > 1 else []
    repaired["annotation_id"] = annotation_id

    raw_images = repaired.get("annotations_image")
    if not isinstance(raw_images, list):
        raw_images = []
    image_by_path = {
        item.get("image_path"): item
        for item in raw_images
        if isinstance(item, dict) and isinstance(item.get("image_path"), str)
    }
    repaired_images = []
    for row in rows:
        annotation = dict(image_by_path.get(row["image_path"], {}))
        annotation.setdefault("image_path", row["image_path"])
        annotation.setdefault("vehicle_id", vehicle_id)
        annotation.setdefault("source_dataset", "cvpair")
        annotation.setdefault("camera_id", row["camera_id"])
        annotation.setdefault("view_source", row["view_source"])
        annotation.setdefault("platform_type", row["platform_type"])
        annotation.setdefault("target_size", row["target_size"])
        annotation.setdefault("small_target", row["small_target"])
        annotation.setdefault("color", annotation_id.get("color", "uncertain"))
        annotation.setdefault("vehicle_type", annotation_id.get("vehicle_type", "uncertain"))
        annotation.setdefault("body_profile", annotation_id.get("body_profile", "uncertain"))
        img_zh, img_en, img_zh_alt, img_en_alt = make_dense_image_captions(annotation)
        annotation["description_zh"] = img_zh
        annotation["description_en"] = img_en
        annotation["description_zh_variants"] = [img_zh_alt] if caption_variants > 1 else []
        annotation["description_en_variants"] = [img_en_alt] if caption_variants > 1 else []
        repaired_images.append(annotation)
    repaired["annotations_image"] = repaired_images
    repaired["qa_notes"] = list(
        dict.fromkeys(ensure_list(repaired.get("qa_notes")) + ["local_caption_density_repair"])
    )
    return repaired


def classify_pixel_color(r: int, g: int, b: int) -> str:
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    hue = h * 360.0
    if s < 0.16:
        if v > 0.78:
            return "white"
        if v < 0.22:
            return "black"
        if v > 0.58:
            return "silver"
        return "gray"
    if hue < 18 or hue >= 345:
        return "red"
    if hue < 45:
        return "brown"
    if hue < 75:
        return "yellow"
    if hue < 165:
        return "green"
    if hue < 255:
        return "blue"
    if hue < 310:
        return "other"
    return "red"


def estimate_color(image_path: Path) -> dict[str, Any]:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((160, 160))
        width, height = img.size
        left = int(width * 0.15)
        top = int(height * 0.15)
        right = max(left + 1, int(width * 0.85))
        bottom = max(top + 1, int(height * 0.85))
        img = img.crop((left, top, right, bottom))
        pixels = list(img.getdata())

    counts: defaultdict[str, int] = defaultdict(int)
    for r, g, b in pixels:
        counts[classify_pixel_color(r, g, b)] += 1

    total = sum(counts.values()) or 1
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    color, count = ranked[0]
    ratio = count / total
    if ratio < 0.32:
        color = "uncertain"
    confidence = round(min(0.82, max(0.35, ratio)), 3)
    return {
        "color": color,
        "confidence": confidence,
        "distribution": {
            key: round(value / total, 3)
            for key, value in ranked[:6]
        },
    }


def visible_parts_for_view(view_source: str) -> list[str]:
    if view_source == "uav":
        return ["roof", "windshield", "hood", "rear_window"]
    return ["windshield", "side_windows", "side_body", "wheels"]


def orientation_for_view(view_source: str) -> str:
    if view_source == "uav":
        return "top"
    return "uncertain"


def image_description(color: str, row: dict[str, Any]) -> tuple[str, str]:
    color_zh = COLOR_ZH.get(color, "颜色不确定")
    color_en = COLOR_EN.get(color, "uncertain-color")
    article = "an" if color_en[0].lower() in {"a", "e", "i", "o", "u"} else "a"
    if row["view_source"] == "uav":
        zh = f"空中俯视下的{color_zh}车辆，可见车顶、挡风玻璃和车身轮廓；车型细节需人工确认。"
        en = (
            f"{article.capitalize()} {color_en} vehicle in an aerial top view, with the roof, "
            "windshield, and body outline visible; fine-grained type details require review."
        )
    else:
        zh = f"地面视角下的{color_zh}车辆，可见挡风玻璃、侧窗和车身侧面；车型细节需人工确认。"
        en = (
            f"{article.capitalize()} {color_en} vehicle in a ground-view crop, with the windshield, "
            "side windows, and side body visible; fine-grained type details require review."
        )
    return zh, en


def id_description(color: str) -> tuple[str, str]:
    color_zh = COLOR_ZH.get(color, "颜色不确定")
    color_en = COLOR_EN.get(color, "uncertain-color")
    if color == "uncertain":
        return (
            "一辆车辆，跨视角稳定颜色和车型仍需人工确认。",
            "A vehicle whose stable color and vehicle type still require manual confirmation.",
        )
    return (
        f"一辆以{color_zh}外观为主的车辆，跨视角可见车身轮廓；具体车型和细节需人工复核。",
        (
            f"A primarily {color_en} vehicle with a visible body outline across views; "
            "the exact vehicle type and fine details require manual review."
        ),
    )


def heuristic_payload(vehicle_id: str, rows: list[dict[str, Any]], dataset_root: Path) -> dict[str, Any]:
    image_annotations = []
    color_votes: defaultdict[str, float] = defaultdict(float)
    color_details: dict[str, Any] = {}

    for idx, row in enumerate(rows, start=1):
        image_path = dataset_root / row["image_path"]
        color_result = estimate_color(image_path)
        color = color_result["color"]
        color_votes[color] += color_result["confidence"]
        color_details[row["image_path"]] = color_result
        uncertain_fields = ["vehicle_type"]
        if color == "uncertain" or color_result["confidence"] < 0.5:
            uncertain_fields.append("color")
        if row["small_target"]:
            uncertain_fields.append("small_target")
        description_zh, description_en = image_description(color, row)
        qa_status = "manual_review" if uncertain_fields else "auto_labeled"
        image_annotations.append(
            {
                "image_index": idx,
                "image_path": row["image_path"],
                "vehicle_id": vehicle_id,
                "source_dataset": "cvpair",
                "camera_id": row["camera_id"],
                "view_source": row["view_source"],
                "platform_type": row["platform_type"],
                "color": color,
                "vehicle_type": "uncertain",
                "orientation": orientation_for_view(row["view_source"]),
                "visible_parts": visible_parts_for_view(row["view_source"]),
                "occlusion": "uncertain",
                "target_size": row["target_size"],
                "small_target": row["small_target"],
                "scene_context": "vehicle_crop_context",
                "weather": None,
                "illumination": "uncertain",
                "background_distractors": {
                    "neighbor_vehicle_count": None,
                    "occluders": [],
                    "complexity": "uncertain",
                },
                "description_zh": description_zh,
                "description_en": description_en,
                "confidence": color_result["confidence"],
                "uncertain_fields": uncertain_fields,
                "qa_status": qa_status,
            }
        )

    stable_color = "uncertain"
    if color_votes:
        stable_color = max(color_votes.items(), key=lambda item: item[1])[0]
        non_uncertain_total = sum(
            value for key, value in color_votes.items() if key != "uncertain"
        )
        if stable_color == "uncertain" or non_uncertain_total < 0.5:
            stable_color = "uncertain"

    description_zh, description_en = id_description(stable_color)
    uncertain_attributes = ["vehicle_type", "occlusion"]
    if stable_color == "uncertain":
        uncertain_attributes.append("color")
    return {
        "annotation_id": {
            "dataset": "tag_vr",
            "vehicle_id": vehicle_id,
            "source_datasets": ["cvpair"],
            "identity_scope": "global",
            "identity_confidence": "global_id_verified",
            "description_zh": description_zh,
            "description_en": description_en,
            "color": stable_color,
            "vehicle_type": "uncertain",
            "body_profile": "uncertain",
            "roof_features": [],
            "window_features": [],
            "cargo_or_rear_structure": "uncertain",
            "special_marks": [],
            "stable_attributes": [] if stable_color == "uncertain" else [f"{stable_color}_body"],
            "uncertain_attributes": uncertain_attributes,
            "qa_status": "manual_review",
        },
        "annotations_image": image_annotations,
        "qa_notes": [
            "heuristic_local_color_estimation",
            "vehicle_type_requires_manual_or_vlm_review",
        ],
        "heuristic_color_details": color_details,
    }


def normalize_image_annotations(
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
    vehicle_id: str,
    caption_variants: int = 2,
) -> list[dict[str, Any]]:
    by_path: dict[str, dict[str, Any]] = {}
    by_index: dict[int, dict[str, Any]] = {}
    raw_annotations = payload.get("annotations_image") or []
    if not isinstance(raw_annotations, list):
        raw_annotations = []
    for ann in raw_annotations:
        if not isinstance(ann, dict):
            continue
        path = ann.get("image_path")
        if isinstance(path, str):
            by_path[path] = ann
        idx = ann.get("image_index")
        if isinstance(idx, int):
            by_index[idx] = ann

    normalized_rows = []
    for idx, row in enumerate(rows, start=1):
        ann = by_path.get(row["image_path"]) or by_index.get(idx) or {}
        uncertain_fields = ensure_list(ann.get("uncertain_fields"))
        qa_status = ann.get("qa_status", "auto_labeled")
        description_zh = ensure_scalar(ann.get("description_zh"), default="")
        description_en = ensure_scalar(ann.get("description_en"), default="")
        if not description_zh or not description_en:
            qa_status = "manual_review"
            if "description" not in uncertain_fields:
                uncertain_fields.append("description")
        normalized = {
                "dataset": "tag_vr",
                "image_path": row["image_path"],
                "vehicle_id": vehicle_id,
                "source_dataset": "cvpair",
                "camera_id": row["camera_id"],
                "view_source": row["view_source"],
                "platform_type": row["platform_type"],
                "color": ensure_scalar(ann.get("color")),
                "vehicle_type": ensure_scalar(ann.get("vehicle_type")),
                "orientation": ensure_scalar(ann.get("orientation")),
                "visible_parts": ensure_list(ann.get("visible_parts")),
                "occlusion": ensure_scalar(ann.get("occlusion")),
                "target_size": row["target_size"],
                "small_target": row["small_target"],
                "scene_context": ensure_scalar(ann.get("scene_context")),
                "weather": normalize_optional_string(ann.get("weather")),
                "illumination": normalize_optional_string(ann.get("illumination")),
                "background_distractors": ann.get(
                    "background_distractors",
                    {
                        "neighbor_vehicle_count": None,
                        "occluders": [],
                        "complexity": "uncertain",
                    },
                ),
                "description_zh": description_zh,
                "description_en": description_en,
                "caption_claims": ann.get("caption_claims")
                if isinstance(ann.get("caption_claims"), dict)
                else {},
                "confidence": normalize_confidence(ann.get("confidence")),
                "uncertain_fields": uncertain_fields,
                "review_reasons": ensure_list(ann.get("review_reasons")),
                "qa_status": ensure_scalar(qa_status, default="auto_labeled"),
                "source_split": row.get("source_split"),
            }
        normalized["description_zh_variants"] = normalize_caption_variants(
            ann.get("description_zh_variants"),
            canonical=description_zh,
            total_count=caption_variants,
        )
        normalized["description_en_variants"] = normalize_caption_variants(
            ann.get("description_en_variants"),
            canonical=description_en,
            total_count=caption_variants,
        )
        normalized_rows.append(normalized)
    return normalized_rows


def preview_caption_length_qa(
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
    vehicle_id: str,
    caption_variants: int,
) -> list[dict[str, Any]]:
    temp_id_row = normalize_id_annotation(
        payload,
        vehicle_id,
        caption_variants=caption_variants,
    )
    temp_image_rows = normalize_image_annotations(
        payload,
        rows,
        vehicle_id,
        caption_variants=caption_variants,
    )
    return apply_caption_length_qa(
        [temp_id_row],
        temp_image_rows,
        expected_caption_variants=caption_variants,
    )


def mock_payload(vehicle_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    image_annotations = []
    for idx, row in enumerate(rows, start=1):
        if row["view_source"] == "uav":
            orientation = "top"
            visible_parts = ["roof", "windshield"]
            zh_view = "空中俯视"
            en_view = "aerial top view"
        else:
            orientation = "front_left"
            visible_parts = ["windshield", "side_windows", "side_body"]
            zh_view = "地面近景"
            en_view = "ground close view"
        image_annotations.append(
            {
                "image_index": idx,
                "image_path": row["image_path"],
                "vehicle_id": vehicle_id,
                "source_dataset": "cvpair",
                "camera_id": row["camera_id"],
                "view_source": row["view_source"],
                "platform_type": row["platform_type"],
                "color": "uncertain",
                "vehicle_type": "uncertain",
                "orientation": orientation,
                "visible_parts": visible_parts,
                "occlusion": "uncertain",
                "target_size": row["target_size"],
                "small_target": row["small_target"],
                "scene_context": "parking_or_road_context",
                "weather": None,
                "illumination": "uncertain",
                "background_distractors": {
                    "neighbor_vehicle_count": None,
                    "occluders": [],
                    "complexity": "uncertain",
                },
                "description_zh": f"{zh_view}下的车辆 crop，颜色和车型需人工确认。",
                "description_en": f"A vehicle crop in {en_view}; color and type require manual confirmation.",
                "confidence": 0.25,
                "uncertain_fields": ["color", "vehicle_type", "occlusion"],
                "qa_status": "manual_review",
            }
        )
    return {
        "annotation_id": {
            "dataset": "tag_vr",
            "vehicle_id": vehicle_id,
            "source_datasets": ["cvpair"],
            "identity_scope": "global",
            "identity_confidence": "global_id_verified",
            "description_zh": "一辆车辆，跨视角稳定颜色和车型需要人工确认。",
            "description_en": "A vehicle whose stable color and type require manual confirmation.",
            "color": "uncertain",
            "vehicle_type": "uncertain",
            "body_profile": "uncertain",
            "roof_features": [],
            "window_features": [],
            "cargo_or_rear_structure": "uncertain",
            "special_marks": [],
            "stable_attributes": [],
            "uncertain_attributes": ["color", "vehicle_type"],
            "qa_status": "manual_review",
        },
        "annotations_image": image_annotations,
        "qa_notes": ["mock output only"],
    }


def merge_perception_payloads(
    base: dict[str, Any],
    supplement: dict[str, Any],
) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    base_by_path = {
        item.get("image_path"): item
        for item in merged.get("images", [])
        if isinstance(item, dict)
    }
    for supplement_image in supplement.get("images", []):
        if not isinstance(supplement_image, dict):
            continue
        image_path = supplement_image.get("image_path")
        if image_path not in base_by_path:
            merged.setdefault("images", []).append(copy.deepcopy(supplement_image))
            base_by_path[image_path] = merged["images"][-1]
            continue
        target = base_by_path[image_path]
        records = {
            (item.get("attribute"), item.get("value")): item
            for item in target.get("attributes", [])
            if isinstance(item, dict)
        }
        for record in supplement_image.get("attributes", []):
            if not isinstance(record, dict):
                continue
            key = (record.get("attribute"), record.get("value"))
            existing = records.get(key)
            if existing is None:
                target.setdefault("attributes", []).append(copy.deepcopy(record))
                records[key] = target["attributes"][-1]
            elif normalize_confidence(record.get("confidence")) >= normalize_confidence(
                existing.get("confidence")
            ):
                existing.update(copy.deepcopy(record))
        target["quality_notes"] = list(
            dict.fromkeys(
                ensure_list(target.get("quality_notes"))
                + ensure_list(supplement_image.get("quality_notes"))
            )
        )
    merged["schema_issues"] = list(
        dict.fromkeys(
            ensure_list(base.get("schema_issues"))
            + ensure_list(supplement.get("schema_issues"))
        )
    )
    evidence_counter = 0
    for image_index, image in enumerate(merged.get("images", []), start=1):
        for record in image.get("attributes", []):
            evidence_counter += 1
            record["evidence_id"] = f"e{image_index:02d}_{evidence_counter:04d}"
    return merged


def merge_caption_output(
    base_payload: dict[str, Any],
    caption_output: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(base_payload)
    annotation_id = caption_output.get("annotation_id")
    annotations_image = caption_output.get("annotations_image")
    if isinstance(annotation_id, dict):
        merged["annotation_id"] = annotation_id
    if isinstance(annotations_image, list):
        merged["annotations_image"] = annotations_image
    merged["new_observations"] = ensure_list(caption_output.get("new_observations"))
    return merged


def apply_v4_automatic_decision(
    payload: dict[str, Any],
    local_qa: dict[str, Any],
    claim_audit: dict[str, Any],
) -> None:
    passed = local_qa.get("status") == "pass" and claim_audit.get("status") == "pass"
    status = "auto_labeled" if passed else "manual_review"
    reasons = []
    if local_qa.get("status") != "pass":
        reasons.append("local_schema_length_or_fact_coverage_qa")
    if claim_audit.get("status") != "pass":
        reasons.append("vlm_b_claim_audit")
    annotation_id = payload.get("annotation_id")
    if isinstance(annotation_id, dict):
        annotation_id["qa_status"] = status
        annotation_id["review_reasons"] = reasons
    for annotation in payload.get("annotations_image", []):
        if isinstance(annotation, dict):
            annotation["qa_status"] = status
            annotation["review_reasons"] = reasons


def run_job_v4(
    vehicle_id: str,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    api_key: str,
    audit_api_key: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    family_a, family_b, audit_base_url = validate_v4_api_configuration(args)
    dataset_root = args.dataset_root.resolve()
    raw_stages: list[dict[str, Any]] = []

    perception_content = build_v4_perception_content(
        vehicle_id,
        rows,
        dataset_root,
        args.max_image_side,
        args.jpeg_quality,
    )
    perception_output, raw_stage = call_json_model_stage(
        "1_vlm_a_per_image_perception",
        args.model,
        args.base_url,
        api_key,
        PERCEPTION_SYSTEM_PROMPT,
        perception_content,
        args,
    )
    raw_stages.append(raw_stage)
    perception = normalize_perception_payload(perception_output, rows)

    def adjudicate_consensus(round_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
        draft = build_program_consensus_draft(
            perception,
            confidence_threshold=args.evidence_confidence_threshold,
        )
        consensus_content = build_v4_consensus_content(vehicle_id, perception, draft)
        adjudication, consensus_stage = call_json_model_stage(
            f"2_vlm_a_consensus_adjudication_round_{round_index}",
            args.model,
            args.base_url,
            api_key,
            CONSENSUS_SYSTEM_PROMPT,
            consensus_content,
            args,
        )
        raw_stages.append(consensus_stage)
        consensus, normalized = finalize_cross_view_consensus(
            perception,
            adjudication,
            confidence_threshold=args.evidence_confidence_threshold,
        )
        raw_stages.append(
            {
                "stage": f"2_program_consensus_finalization_round_{round_index}",
                "program_draft": draft,
                "vlm_a_adjudication": adjudication,
                "final_consensus": consensus,
            }
        )
        return consensus, normalized

    consensus, perception = adjudicate_consensus(0)
    payload: dict[str, Any] = {
        "perception": perception,
        "cross_view_consensus": consensus,
        "evidence_confidence_threshold": args.evidence_confidence_threshold,
        "new_observations": [],
    }

    observation_round = 0
    while True:
        caption_content = build_v4_caption_content(
            vehicle_id,
            rows,
            dataset_root,
            payload,
            args.max_image_side,
            args.jpeg_quality,
            args.caption_variants,
            args.id_coverage_threshold,
            args.image_coverage_threshold,
        )
        caption_output, caption_stage = call_json_model_stage(
            f"3_vlm_a_fact_locked_caption_round_{observation_round}",
            args.model,
            args.base_url,
            api_key,
            FACT_LOCKED_CAPTION_SYSTEM_PROMPT,
            caption_content,
            args,
        )
        raw_stages.append(caption_stage)
        payload = merge_caption_output(payload, caption_output)
        new_observations = ensure_list(payload.get("new_observations"))
        if not new_observations or observation_round >= args.max_new_observation_rounds:
            break
        observation_round += 1
        supplement_content = build_v4_perception_content(
            vehicle_id,
            rows,
            dataset_root,
            args.max_image_side,
            args.jpeg_quality,
            requested_observations=new_observations,
        )
        supplement_output, supplement_stage = call_json_model_stage(
            f"1_vlm_a_perception_reentry_round_{observation_round}",
            args.model,
            args.base_url,
            api_key,
            PERCEPTION_SYSTEM_PROMPT,
            supplement_content,
            args,
        )
        raw_stages.append(supplement_stage)
        supplement = normalize_perception_payload(supplement_output, rows)
        perception = merge_perception_payloads(perception, supplement)
        consensus, perception = adjudicate_consensus(observation_round)
        payload["perception"] = perception
        payload["cross_view_consensus"] = consensus

    payload["fact_tables"] = build_authorized_fact_tables(
        payload,
        rows,
        confidence_threshold=args.evidence_confidence_threshold,
    )
    first_local_qa = run_v4_local_qa(
        payload,
        rows,
        vehicle_id,
        args.caption_variants,
        args.evidence_confidence_threshold,
        args.id_coverage_threshold,
        args.image_coverage_threshold,
        args.uncertainty_coverage_threshold,
    )
    raw_stages.append({"stage": "4_local_schema_length_fact_coverage_qa", **first_local_qa})

    audit_content = build_v4_claim_audit_content(
        vehicle_id,
        rows,
        dataset_root,
        payload,
        first_local_qa,
        args.max_image_side,
        args.jpeg_quality,
        args.caption_variants,
        audit_round=1,
    )
    first_audit_output, first_audit_stage = call_json_model_stage(
        "5_vlm_b_claim_audit_round_1",
        args.model_b,
        audit_base_url,
        audit_api_key,
        CLAIM_AUDIT_SYSTEM_PROMPT,
        audit_content,
        args,
    )
    raw_stages.append(first_audit_stage)
    first_claim_audit = normalize_claim_audit(
        first_audit_output,
        rows,
        vehicle_id,
        args.caption_variants,
        audit_round=1,
    )

    repair_triggered = (
        first_local_qa.get("status") != "pass"
        or first_claim_audit.get("status") != "pass"
    )
    final_local_qa = first_local_qa
    final_claim_audit = first_claim_audit
    claim_audit_rounds = [first_claim_audit]
    if repair_triggered:
        repair_content = build_v4_audit_repair_content(
            vehicle_id,
            rows,
            payload,
            first_local_qa,
            first_claim_audit,
            args.caption_variants,
        )
        repair_output, repair_stage = call_json_model_stage(
            "6_vlm_a_repair_from_audit_issues",
            args.model,
            args.base_url,
            api_key,
            AUDIT_REPAIR_SYSTEM_PROMPT,
            repair_content,
            args,
        )
        raw_stages.append(repair_stage)
        payload = merge_caption_output(payload, repair_output)
        final_local_qa = run_v4_local_qa(
            payload,
            rows,
            vehicle_id,
            args.caption_variants,
            args.evidence_confidence_threshold,
            args.id_coverage_threshold,
            args.image_coverage_threshold,
            args.uncertainty_coverage_threshold,
        )
        raw_stages.append({"stage": "7_local_qa_after_vlm_a_repair", **final_local_qa})
        final_audit_content = build_v4_claim_audit_content(
            vehicle_id,
            rows,
            dataset_root,
            payload,
            final_local_qa,
            args.max_image_side,
            args.jpeg_quality,
            args.caption_variants,
            audit_round=2,
        )
        final_audit_output, final_audit_stage = call_json_model_stage(
            "8_vlm_b_final_claim_reaudit",
            args.model_b,
            audit_base_url,
            audit_api_key,
            CLAIM_AUDIT_SYSTEM_PROMPT,
            final_audit_content,
            args,
        )
        raw_stages.append(final_audit_stage)
        final_claim_audit = normalize_claim_audit(
            final_audit_output,
            rows,
            vehicle_id,
            args.caption_variants,
            audit_round=2,
        )
        claim_audit_rounds.append(final_claim_audit)

    payload["local_qa"] = final_local_qa
    payload["claim_audit"] = final_claim_audit
    payload["claim_audit_rounds"] = claim_audit_rounds
    payload["semantic_audit"] = semantic_audit_from_claim_audit(final_claim_audit, rows)
    payload["fact_tables"] = build_authorized_fact_tables(
        payload,
        rows,
        confidence_threshold=args.evidence_confidence_threshold,
    )
    apply_v4_automatic_decision(payload, final_local_qa, final_claim_audit)
    raw = {
        "vehicle_id": vehicle_id,
        "backend": "api",
        "pipeline": "staged_claim_audit",
        "annotation_method_version": ANNOTATION_METHOD_VERSION,
        "prompt_version": PROMPT_VERSION,
        "model_a": args.model,
        "model_a_family": family_a,
        "model_b": args.model_b,
        "model_b_family": family_b,
        "base_url_a": args.base_url,
        "base_url_b": audit_base_url,
        "selected_images": [row["image_path"] for row in rows],
        "new_observation_rounds": observation_round,
        "repair_triggered": repair_triggered,
        "final_local_qa_status": final_local_qa.get("status"),
        "final_claim_audit_status": final_claim_audit.get("status"),
        "stages": raw_stages,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return payload, raw


def run_job_legacy(
    vehicle_id: str,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    api_key: Optional[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    backend = "mock" if args.mock else args.backend
    if backend == "mock":
        payload = mock_payload(vehicle_id, rows)
        raw = {
            "vehicle_id": vehicle_id,
            "backend": "mock",
            "pipeline": args.pipeline,
            "model": "mock",
            "annotation_method_version": ANNOTATION_METHOD_VERSION,
            "prompt_version": PROMPT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "selected_images": [row["image_path"] for row in rows],
            "response_text": json.dumps(payload, ensure_ascii=False),
            "usage": None,
        }
        return payload, raw

    if backend == "heuristic":
        payload = heuristic_payload(vehicle_id, rows, args.dataset_root.resolve())
        raw = {
            "vehicle_id": vehicle_id,
            "backend": "heuristic",
            "pipeline": args.pipeline,
            "model": "local_color_heuristic",
            "annotation_method_version": ANNOTATION_METHOD_VERSION,
            "prompt_version": PROMPT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "selected_images": [row["image_path"] for row in rows],
            "response_text": json.dumps(payload, ensure_ascii=False),
            "usage": None,
        }
        return payload, raw

    if not api_key:
        raise SystemExit(
            f"Missing API key. Set {args.api_key_env} or OPENAI_API_KEY, "
            "or use --backend heuristic/--mock."
        )

    content = build_user_content(
        vehicle_id,
        rows,
        args.dataset_root.resolve(),
        max_image_side=args.max_image_side,
        jpeg_quality=args.jpeg_quality,
        pipeline=args.pipeline,
        caption_variants=args.caption_variants,
        semantic_match_threshold=args.semantic_match_threshold,
    )
    request_payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "response_format": {"type": "json_object"},
    }
    initial_response = post_chat_completion(
        base_url=args.base_url,
        api_key=api_key,
        payload=request_payload,
        timeout=args.timeout,
        retries=args.retries,
    )
    initial_text = extract_message_text(initial_response)
    initial_payload = parse_json_object(initial_text)

    if args.pipeline == "cot_audit":
        audit_content = build_audit_user_content(
            vehicle_id,
            rows,
            args.dataset_root.resolve(),
            initial_payload=initial_payload,
            max_image_side=args.max_image_side,
            jpeg_quality=args.jpeg_quality,
            caption_variants=args.caption_variants,
            semantic_match_threshold=args.semantic_match_threshold,
        )
        audit_request_payload = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": AUDIT_SYSTEM_PROMPT},
                {"role": "user", "content": audit_content},
            ],
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "response_format": {"type": "json_object"},
        }
        audit_response = post_chat_completion(
            base_url=args.base_url,
            api_key=api_key,
            payload=audit_request_payload,
            timeout=args.timeout,
            retries=args.retries,
        )
        audit_text = extract_message_text(audit_response)
        audit_payload = parse_json_object(audit_text)
        refined = audit_payload.get("refined_annotation")
        payload = dict(refined) if isinstance(refined, dict) else dict(initial_payload)
        payload["perception"] = initial_payload.get("perception", {})
        payload["cross_view_consensus"] = initial_payload.get("cross_view_consensus", {})
        payload["semantic_audit"] = audit_payload.get("semantic_audit", {})
        payload["qa_notes"] = ensure_list(initial_payload.get("qa_notes"))
        raw_stages = [
            {
                "stage": "perception_and_generation",
                "response_text": initial_text,
                "usage": initial_response.get("usage"),
                "finish_reason": (initial_response.get("choices") or [{}])[0].get("finish_reason"),
            },
            {
                "stage": "visual_audit_and_refinement",
                "response_text": audit_text,
                "usage": audit_response.get("usage"),
                "finish_reason": (audit_response.get("choices") or [{}])[0].get("finish_reason"),
            },
        ]
        response_text = audit_text
        usage = {
            "perception_and_generation": initial_response.get("usage"),
            "visual_audit_and_refinement": audit_response.get("usage"),
        }
        finish_reason = (audit_response.get("choices") or [{}])[0].get("finish_reason")
        audit_instruction = audit_content[0]["text"]
    else:
        payload = initial_payload
        raw_stages = [
            {
                "stage": "single_pass",
                "response_text": initial_text,
                "usage": initial_response.get("usage"),
                "finish_reason": (initial_response.get("choices") or [{}])[0].get("finish_reason"),
            }
        ]
        response_text = initial_text
        usage = initial_response.get("usage")
        finish_reason = (initial_response.get("choices") or [{}])[0].get("finish_reason")
        audit_instruction = None

    caption_repair_instruction = None
    caption_repair_triggered = False
    caption_repair_error = None
    caption_qa_before_repair: list[dict[str, Any]] = []
    local_caption_repair_triggered = False
    caption_qa_before_local_repair: list[dict[str, Any]] = []
    caption_qa_after_local_repair: list[dict[str, Any]] = []
    if args.repair_caption_length:
        caption_qa_before_repair = preview_caption_length_qa(
            payload,
            rows,
            vehicle_id,
            caption_variants=args.caption_variants,
        )
        if any(row["status"] != "pass" for row in caption_qa_before_repair):
            caption_repair_triggered = True
            repair_content = build_caption_repair_user_content(
                vehicle_id,
                rows,
                current_payload=payload,
                caption_qa_rows=caption_qa_before_repair,
                caption_variants=args.caption_variants,
            )
            caption_repair_instruction = repair_content[0]["text"]
            repair_request_payload = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": CAPTION_REPAIR_SYSTEM_PROMPT},
                    {"role": "user", "content": repair_content},
                ],
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "response_format": {"type": "json_object"},
            }
            try:
                repair_response = post_chat_completion(
                    base_url=args.base_url,
                    api_key=api_key,
                    payload=repair_request_payload,
                    timeout=args.timeout,
                    retries=args.retries,
                )
                repair_text = extract_message_text(repair_response)
                repair_payload = parse_json_object(repair_text)
                refined = repair_payload.get("refined_annotation")
                candidate_payload = refined if isinstance(refined, dict) else repair_payload
                if isinstance(candidate_payload, dict) and (
                    "annotation_id" in candidate_payload or "annotations_image" in candidate_payload
                ):
                    repaired_payload = dict(candidate_payload)
                    repaired_payload["perception"] = payload.get("perception", {})
                    repaired_payload["cross_view_consensus"] = payload.get("cross_view_consensus", {})
                    repaired_payload["semantic_audit"] = repair_payload.get(
                        "semantic_audit",
                        payload.get("semantic_audit", {}),
                    )
                    repaired_payload["qa_notes"] = list(
                        dict.fromkeys(
                            ensure_list(payload.get("qa_notes"))
                            + ensure_list(repair_payload.get("qa_notes"))
                        )
                    )
                    payload = repaired_payload
                raw_stages.append(
                    {
                        "stage": "caption_length_repair",
                        "response_text": repair_text,
                        "usage": repair_response.get("usage"),
                        "finish_reason": (repair_response.get("choices") or [{}])[0].get("finish_reason"),
                        "caption_qa_failures": compact_caption_qa_failures(caption_qa_before_repair),
                    }
                )
                response_text = repair_text
                finish_reason = (repair_response.get("choices") or [{}])[0].get("finish_reason")
                if isinstance(usage, dict) and not {
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                }.intersection(usage):
                    usage["caption_length_repair"] = repair_response.get("usage")
                else:
                    usage = {
                        "previous": usage,
                        "caption_length_repair": repair_response.get("usage"),
                    }
            except Exception as exc:  # noqa: BLE001 - keep original annotation auditable.
                caption_repair_error = repr(exc)
                raw_stages.append(
                    {
                        "stage": "caption_length_repair",
                        "error": caption_repair_error,
                        "caption_qa_failures": compact_caption_qa_failures(caption_qa_before_repair),
                    }
                )
        caption_qa_before_local_repair = preview_caption_length_qa(
            payload,
            rows,
            vehicle_id,
            caption_variants=args.caption_variants,
        )
        if any(row["status"] != "pass" for row in caption_qa_before_local_repair):
            local_caption_repair_triggered = True
            payload = apply_local_caption_density_repair(
                payload,
                rows,
                vehicle_id,
                caption_variants=args.caption_variants,
            )
            caption_qa_after_local_repair = preview_caption_length_qa(
                payload,
                rows,
                vehicle_id,
                caption_variants=args.caption_variants,
            )
            raw_stages.append(
                {
                    "stage": "local_caption_density_repair",
                    "caption_qa_failures_before": compact_caption_qa_failures(
                        caption_qa_before_local_repair
                    ),
                    "caption_qa_failures_after": compact_caption_qa_failures(
                        caption_qa_after_local_repair
                    ),
                    "note": (
                        "Deterministic fallback using parsed structured fields; "
                        "no additional image upload or new visual facts."
                    ),
                }
            )

    raw = {
        "vehicle_id": vehicle_id,
        "backend": "api",
        "pipeline": args.pipeline,
        "model": args.model,
        "base_url": args.base_url,
        "annotation_method_version": ANNOTATION_METHOD_VERSION,
        "prompt_version": PROMPT_VERSION,
        "system_prompt": SYSTEM_PROMPT,
        "user_instruction": content[0]["text"],
        "audit_system_prompt": AUDIT_SYSTEM_PROMPT if args.pipeline == "cot_audit" else None,
        "audit_user_instruction": audit_instruction,
        "caption_length_repair_enabled": args.repair_caption_length,
        "caption_length_repair_triggered": caption_repair_triggered,
        "caption_length_repair_error": caption_repair_error,
        "local_caption_density_repair_triggered": local_caption_repair_triggered,
        "caption_repair_system_prompt": CAPTION_REPAIR_SYSTEM_PROMPT if caption_repair_triggered else None,
        "caption_repair_user_instruction": caption_repair_instruction,
        "caption_length_qa_before_repair": compact_caption_qa_failures(caption_qa_before_repair),
        "caption_length_qa_before_local_repair": compact_caption_qa_failures(
            caption_qa_before_local_repair
        ),
        "caption_length_qa_after_local_repair": compact_caption_qa_failures(
            caption_qa_after_local_repair
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selected_images": [row["image_path"] for row in rows],
        "response_text": response_text,
        "stages": raw_stages,
        "usage": usage,
        "finish_reason": finish_reason,
    }
    return payload, raw


def run_job(
    vehicle_id: str,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    api_key: Optional[str],
    audit_api_key: Optional[str] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    backend = "mock" if args.mock else args.backend
    if backend == "api" and args.pipeline == "staged_claim_audit":
        if not api_key:
            raise SystemExit(
                f"Missing VLM A API key. Set {args.api_key_env} or OPENAI_API_KEY."
            )
        return run_job_v4(
            vehicle_id,
            rows,
            args,
            api_key,
            audit_api_key or api_key,
        )
    return run_job_legacy(vehicle_id, rows, args, api_key)


def write_report(
    output_dir: Path,
    backend: str,
    model: str,
    jobs: list[tuple[str, list[dict[str, Any]]]],
    id_rows: list[dict[str, Any]],
    image_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    caption_qa_rows: list[dict[str, Any]],
    semantic_qa_rows: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
    schema_qa_rows: list[dict[str, Any]],
    fact_coverage_qa_rows: list[dict[str, Any]],
    claim_audit_summary_rows: list[dict[str, Any]],
    pipeline: str,
    caption_variants: int,
    mock: bool,
) -> None:
    low_conf = [
        row for row in image_rows if float(row.get("confidence") or 0.0) < 0.5
    ]
    manual_review = [
        row for row in image_rows if row.get("qa_status") == "manual_review"
    ]
    manual_review_ids = [
        row for row in id_rows if row.get("qa_status") == "manual_review"
    ]
    usage_rows = [row.get("usage") for row in raw_rows if row.get("usage")]
    caption_qa_pass = [row for row in caption_qa_rows if row["status"] == "pass"]
    caption_qa_fail = [row for row in caption_qa_rows if row["status"] != "pass"]
    id_caption_fail = [row for row in caption_qa_fail if row["level"] == "id"]
    image_caption_fail = [row for row in caption_qa_fail if row["level"] == "image"]
    semantic_qa_pass = [row for row in semantic_qa_rows if row["status"] == "pass"]
    semantic_qa_fail = [row for row in semantic_qa_rows if row["status"] != "pass"]
    semantic_categories: defaultdict[str, int] = defaultdict(int)
    for row in semantic_qa_rows:
        semantic_categories[row["category"]] += 1
    review_required_ids = [row for row in id_rows if row.get("review_required")]
    review_required_images = [row for row in image_rows if row.get("review_required")]
    evidence_qa_pass = [
        row for row in evidence_rows if row.get("evidence_qa", {}).get("status") == "pass"
    ]
    schema_qa_pass = [row for row in schema_qa_rows if row.get("status") == "pass"]
    fact_coverage_qa_pass = [
        row for row in fact_coverage_qa_rows if row.get("status") == "pass"
    ]
    final_claim_audit_summaries = select_final_claim_audit_summaries(
        claim_audit_summary_rows
    )
    claim_audit_pass = [
        row for row in final_claim_audit_summaries if row.get("status") == "pass"
    ]
    report = f"""# CVPair Annotation Smoke Test Report

Generated at: `{datetime.now(timezone.utc).isoformat()}`

- Mode: `{backend}`
- Model: `{model}`
- Annotation method: `{ANNOTATION_METHOD_VERSION}`
- Pipeline: `{pipeline}`
- Caption variants per annotation: {caption_variants}
- Vehicle IDs annotated: {len(id_rows)}
- Image annotations: {len(image_rows)}
- Evidence records: {len(evidence_rows)}
- Evidence QA passed: {len(evidence_qa_pass)} / {len(evidence_rows)}
- Low confidence image annotations: {len(low_conf)}
- Manual review ID annotations: {len(manual_review_ids)}
- Manual review image annotations: {len(manual_review)}
- ID annotations requiring human review: {len(review_required_ids)}
- Image annotations requiring human review: {len(review_required_images)}
- API usage records with usage field: {len(usage_rows)}
- Caption length QA passed: {len(caption_qa_pass)} / {len(caption_qa_rows)}
- ID caption length QA failures: {len(id_caption_fail)}
- Image caption length QA failures: {len(image_caption_fail)}
- Semantic QA passed: {len(semantic_qa_pass)} / {len(semantic_qa_rows)}
- Semantic QA category counts: {json.dumps(dict(sorted(semantic_categories.items())), ensure_ascii=False)}
- Schema QA passed: {len(schema_qa_pass)} / {len(schema_qa_rows)}
- Fact coverage QA passed: {len(fact_coverage_qa_pass)} / {len(fact_coverage_qa_rows)}
- VLM B final claim-audit variants passed: {len(claim_audit_pass)} / {len(final_claim_audit_summaries)}
- VLM B claim-audit summaries across all rounds: {len(claim_audit_summary_rows)}

## Caption Standard

- ID-level: Chinese 60-100 characters, English 35-60 words, each 1-2 sentences.
- Image-level: Chinese 50-90 characters, English 30-55 words, each 1-2 sentences.
- Each annotation contains one canonical caption and {max(0, caption_variants - 1)} natural paraphrase(s).
- Failed rows are automatically marked `manual_review`.

## Semantic QA Standard

- Categories: `match`, `contradictory`, `hallucinatory`, `vacuous`.
- Only `match` rows meeting the configured score threshold pass automatically.
- Query/gallery identities and images require human review even when automatic QA passes.

## V4 Fact And Claim QA

- ID fact coverage must be at least 0.80; image fact coverage must be at least 0.85.
- Required uncertainty reasons must have coverage 1.00.
- Any unsupported fact ID or VLM B claim status other than `supported` fails the row.
- VLM B runs after local QA and uses a different model family without VLM A confidence.

## Selected Jobs

"""
    for vehicle_id, rows in jobs:
        report += f"- `{vehicle_id}`: " + ", ".join(row["image_path"] for row in rows) + "\n"
    if caption_qa_fail:
        report += "\n## Caption Length Failures\n\n"
        for row in caption_qa_fail:
            target = row.get("image_path") or row.get("vehicle_id")
            report += (
                f"- `{row['level']}` `{target}`: "
                f"zh={row['description_zh_characters']}, "
                f"en={row['description_en_words']}, "
                f"issues={', '.join(row['issues'])}\n"
            )
    if semantic_qa_fail:
        report += "\n## Semantic QA Failures\n\n"
        for row in semantic_qa_fail:
            target = row.get("image_path") or row.get("vehicle_id")
            report += (
                f"- `{row['level']}` `{target}`: "
                f"category={row['category']}, score={row['match_score']:.3f}, "
                f"issues={', '.join(str(item) for item in row['field_issues']) or 'none'}\n"
            )
    report += """
## Quick QA Notes

- Verify that ID-level descriptions do not include background identity shortcuts.
- Verify that image-level descriptions only describe visible content.
- Treat `manual_review` and low-confidence rows as not publishable until checked.
"""
    (output_dir / "annotation_smoke_test_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_numeric_args(args)
    backend = "mock" if args.mock else args.backend
    effective_pipeline = args.pipeline if backend == "api" else f"{backend}_local"
    dataset_root = args.dataset_root.resolve()
    metadata_path = args.metadata or (dataset_root / "metadata" / "metadata.jsonl")
    output_dir = args.output_dir or (dataset_root / "annotations" / "cvpair_smoke_test")
    output_dir = output_dir.resolve()

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output dir already has files; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_jsonl(metadata_path)
    jobs = select_jobs(
        metadata=metadata,
        vehicle_ids=args.vehicle_id,
        limit_ids=args.limit_ids,
        images_per_id=args.images_per_id,
        require_both_views=args.require_both_views,
    )

    api_key = os.environ.get(args.api_key_env) or os.environ.get("OPENAI_API_KEY")
    audit_api_key = os.environ.get(args.audit_api_key_env) or api_key
    id_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    semantic_qa_rows: list[dict[str, Any]] = []
    schema_qa_rows: list[dict[str, Any]] = []
    fact_coverage_qa_rows: list[dict[str, Any]] = []
    claim_audit_rows: list[dict[str, Any]] = []
    claim_audit_summary_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for vehicle_id, rows in jobs:
        try:
            payload, raw = run_job(vehicle_id, rows, args, api_key, audit_api_key)
            id_row = normalize_id_annotation(
                payload,
                vehicle_id,
                caption_variants=args.caption_variants,
            )
            job_image_rows = normalize_image_annotations(
                payload,
                rows,
                vehicle_id,
                caption_variants=args.caption_variants,
            )
            job_semantic_qa_rows = build_semantic_qa_rows(
                payload,
                vehicle_id,
                job_image_rows,
                match_threshold=args.semantic_match_threshold,
            )
            apply_semantic_qa(id_row, job_image_rows, job_semantic_qa_rows)
            if backend == "api" and args.pipeline == "staged_claim_audit":
                local_qa = payload.get("local_qa") if isinstance(payload.get("local_qa"), dict) else {}
                job_schema_rows = ensure_list(local_qa.get("schema_qa"))
                job_coverage_rows = ensure_list(local_qa.get("fact_coverage_qa"))
                job_claim_audit = (
                    payload.get("claim_audit")
                    if isinstance(payload.get("claim_audit"), dict)
                    else {"summary_rows": [], "claim_rows": [], "status": "manual_review"}
                )
                apply_v4_quality_qa(
                    id_row,
                    job_image_rows,
                    job_schema_rows,
                    job_coverage_rows,
                    job_claim_audit,
                )
                schema_qa_rows.extend(job_schema_rows)
                fact_coverage_qa_rows.extend(job_coverage_rows)
                audit_rounds = ensure_list(payload.get("claim_audit_rounds")) or [
                    job_claim_audit
                ]
                for audit_round in audit_rounds:
                    if not isinstance(audit_round, dict):
                        continue
                    claim_audit_rows.extend(
                        ensure_list(audit_round.get("claim_rows"))
                    )
                    claim_audit_summary_rows.extend(
                        ensure_list(audit_round.get("summary_rows"))
                    )
            evidence_row = build_annotation_evidence_row(
                payload,
                vehicle_id,
                rows,
                backend=backend,
                pipeline=effective_pipeline,
            )
            apply_evidence_qa(id_row, job_image_rows, evidence_row)
            apply_human_review_policy(
                id_row,
                job_image_rows,
                rows,
                enforce_eval_review=args.human_review_eval,
                train_review_rate=args.train_review_rate,
                review_sampling_seed=args.review_sampling_seed,
            )
            id_rows.append(id_row)
            image_rows.extend(job_image_rows)
            semantic_qa_rows.extend(job_semantic_qa_rows)
            evidence_rows.append(evidence_row)
            raw["parsed_payload"] = payload
            raw_rows.append(raw)
        except Exception as exc:  # noqa: BLE001 - persist auditable failure.
            errors.append(
                {
                    "vehicle_id": vehicle_id,
                    "selected_images": [row["image_path"] for row in rows],
                    "error": repr(exc),
                }
            )
            print(f"Annotation failed for {vehicle_id}: {exc}", file=sys.stderr)

    caption_qa_rows = apply_caption_length_qa(
        id_rows,
        image_rows,
        expected_caption_variants=args.caption_variants,
    )
    write_jsonl(output_dir / "annotations_id.jsonl", id_rows)
    write_jsonl(output_dir / "annotations_image.jsonl", image_rows)
    write_jsonl(output_dir / "annotation_evidence.jsonl", evidence_rows)
    write_jsonl(output_dir / "semantic_qa.jsonl", semantic_qa_rows)
    write_jsonl(output_dir / "schema_qa.jsonl", schema_qa_rows)
    write_jsonl(output_dir / "fact_coverage_qa.jsonl", fact_coverage_qa_rows)
    write_jsonl(output_dir / "claim_audit.jsonl", claim_audit_rows)
    write_jsonl(output_dir / "claim_audit_summary.jsonl", claim_audit_summary_rows)
    write_jsonl(output_dir / "raw_responses.jsonl", raw_rows)
    write_jsonl(output_dir / "errors.jsonl", errors)
    write_jsonl(output_dir / "caption_length_qa.jsonl", caption_qa_rows)
    caption_qa_pass_count = sum(1 for row in caption_qa_rows if row["status"] == "pass")
    semantic_qa_pass_count = sum(1 for row in semantic_qa_rows if row["status"] == "pass")
    evidence_qa_pass_count = sum(
        1
        for row in evidence_rows
        if row.get("evidence_qa", {}).get("status") == "pass"
    )
    schema_qa_pass_count = sum(1 for row in schema_qa_rows if row.get("status") == "pass")
    fact_coverage_qa_pass_count = sum(
        1 for row in fact_coverage_qa_rows if row.get("status") == "pass"
    )
    final_claim_audit_summary_rows = select_final_claim_audit_summaries(
        claim_audit_summary_rows
    )
    claim_audit_pass_count = sum(
        1 for row in final_claim_audit_summary_rows if row.get("status") == "pass"
    )
    model_a_family = args.model_a_family or infer_model_family(args.model)
    model_b_family = args.model_b_family or infer_model_family(args.model_b)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": dataset_root.as_posix(),
        "metadata_path": metadata_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "backend": backend,
        "model": "mock" if backend == "mock" else "local_color_heuristic" if backend == "heuristic" else args.model,
        "model_a": args.model if backend == "api" else None,
        "model_a_family": model_a_family if backend == "api" else None,
        "model_b": args.model_b if backend == "api" and args.pipeline == "staged_claim_audit" else None,
        "model_b_family": model_b_family if backend == "api" and args.pipeline == "staged_claim_audit" else None,
        "annotation_method_version": ANNOTATION_METHOD_VERSION,
        "prompt_version": PROMPT_VERSION,
        "pipeline": effective_pipeline,
        "requested_pipeline": args.pipeline,
        "caption_variants": args.caption_variants,
        "caption_length_repair": bool(
            backend == "api"
            and args.pipeline != "staged_claim_audit"
            and args.repair_caption_length
        ),
        "evidence_confidence_threshold": args.evidence_confidence_threshold,
        "id_coverage_threshold": args.id_coverage_threshold,
        "image_coverage_threshold": args.image_coverage_threshold,
        "uncertainty_coverage_threshold": args.uncertainty_coverage_threshold,
        "semantic_match_threshold": args.semantic_match_threshold,
        "human_review_eval": args.human_review_eval,
        "train_review_rate": args.train_review_rate,
        "review_sampling_seed": args.review_sampling_seed,
        "caption_length_rules": CAPTION_LENGTH_RULES,
        "base_url": args.base_url if backend == "api" else None,
        "audit_base_url": (args.audit_base_url or args.base_url)
        if backend == "api" and args.pipeline == "staged_claim_audit"
        else None,
        "limit_ids": args.limit_ids,
        "images_per_id": args.images_per_id,
        "selected_vehicle_ids": [vehicle_id for vehicle_id, _ in jobs],
        "annotation_id_count": len(id_rows),
        "annotation_image_count": len(image_rows),
        "annotation_evidence_count": len(evidence_rows),
        "evidence_qa_pass_count": evidence_qa_pass_count,
        "evidence_qa_fail_count": len(evidence_rows) - evidence_qa_pass_count,
        "semantic_qa_count": len(semantic_qa_rows),
        "semantic_qa_pass_count": semantic_qa_pass_count,
        "semantic_qa_fail_count": len(semantic_qa_rows) - semantic_qa_pass_count,
        "schema_qa_count": len(schema_qa_rows),
        "schema_qa_pass_count": schema_qa_pass_count,
        "schema_qa_fail_count": len(schema_qa_rows) - schema_qa_pass_count,
        "fact_coverage_qa_count": len(fact_coverage_qa_rows),
        "fact_coverage_qa_pass_count": fact_coverage_qa_pass_count,
        "fact_coverage_qa_fail_count": len(fact_coverage_qa_rows)
        - fact_coverage_qa_pass_count,
        "claim_audit_round_summary_count": len(claim_audit_summary_rows),
        "claim_audit_summary_count": len(final_claim_audit_summary_rows),
        "claim_audit_pass_count": claim_audit_pass_count,
        "claim_audit_fail_count": len(final_claim_audit_summary_rows)
        - claim_audit_pass_count,
        "caption_length_qa_count": len(caption_qa_rows),
        "caption_length_pass_count": caption_qa_pass_count,
        "caption_length_fail_count": len(caption_qa_rows) - caption_qa_pass_count,
        "error_count": len(errors),
    }
    write_json(output_dir / "run_manifest.json", manifest)
    write_report(
        output_dir,
        manifest["backend"],
        manifest["model"],
        jobs,
        id_rows,
        image_rows,
        raw_rows,
        caption_qa_rows,
        semantic_qa_rows,
        evidence_rows,
        schema_qa_rows,
        fact_coverage_qa_rows,
        claim_audit_summary_rows,
        effective_pipeline,
        args.caption_variants,
        args.mock,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
