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
from collections import defaultdict
from datetime import datetime, timezone
import colorsys
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


SYSTEM_PROMPT = """你是 TAG-VR / Text-enhanced Aerial-Ground Vehicle Retrieval 项目的车辆文本标注助手。
你的任务是基于输入车辆 crop 的可见内容，输出可解析 JSON。必须遵守：
1. 只描述图像中可见的车辆属性，不猜品牌、具体型号和真实车牌。
2. ID 级描述只写跨视角稳定车辆属性；背景、停车位、道路纹理、柱体和邻车不能写成身份特征。
3. 图像级描述可记录当前视角、可见部件、遮挡、背景干扰和不确定字段。
4. 小目标、遮挡、颜色难辨、车型难辨时使用 uncertain，并把字段写入 uncertain_fields。
5. 同时输出中文 description_zh 和英文 description_en。
6. 不要输出 Markdown、解释文字或代码块，只输出一个 JSON object。
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
        default=2,
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
        default=os.environ.get("TAG_VR_VLM_MODEL", "gpt-4o-mini"),
        help="OpenAI-compatible VLM model name.",
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
        help="Environment variable containing API key.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2200)
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

    selected: list[dict[str, Any]] = []
    for view in ("ground_camera", "uav"):
        view_rows = sorted(
            by_view.get(view, []),
            key=lambda r: (
                bool(r.get("small_target")),
                r.get("source_split") != "query",
                r.get("image_path", ""),
            ),
        )
        if view_rows:
            selected.append(view_rows[0])

    if len(selected) < images_per_id:
        existing = {row["image_path"] for row in selected}
        rest = sorted(
            [row for row in rows if row["image_path"] not in existing],
            key=lambda r: (bool(r.get("small_target")), r.get("image_path", "")),
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
            }
        )

    schema = {
        "annotation_id": {
            "dataset": "tag_vr",
            "vehicle_id": vehicle_id,
            "source_datasets": ["cvpair"],
            "identity_scope": "global",
            "identity_confidence": "global_id_verified",
            "description_zh": "中文 ID 级稳定描述",
            "description_en": "English stable ID-level description",
            "color": "white|black|gray|silver|red|blue|yellow|green|brown|other|uncertain",
            "vehicle_type": CONTROLLED_ENUMS["vehicle_type"],
            "body_profile": "short controlled phrase or uncertain",
            "roof_features": ["controlled visible roof feature or uncertain"],
            "window_features": ["controlled visible window feature or uncertain"],
            "cargo_or_rear_structure": "short controlled phrase or uncertain",
            "special_marks": ["visible special mark, or empty"],
            "stable_attributes": ["stable vehicle attributes only"],
            "uncertain_attributes": ["fields that are uncertain"],
            "qa_status": "auto_labeled|manual_review",
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
                "vehicle_type": CONTROLLED_ENUMS["vehicle_type"],
                "orientation": CONTROLLED_ENUMS["orientation"],
                "visible_parts": CONTROLLED_ENUMS["visible_parts"],
                "occlusion": CONTROLLED_ENUMS["occlusion"],
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
                "description_zh": "中文图像级可见描述",
                "description_en": "English image-level visible description",
                "confidence": "number from 0 to 1",
                "uncertain_fields": ["field names"],
                "qa_status": "auto_labeled|manual_review",
            }
        ],
        "qa_notes": ["short notes about uncertainty or potential review needs"],
    }

    instruction = {
        "task": "TAG-VR CVPair annotation",
        "vehicle_id": vehicle_id,
        "image_metadata": image_meta,
        "output_schema": schema,
        "controlled_enums": CONTROLLED_ENUMS,
        "important": [
            "Return strict JSON only.",
            "annotations_image length must equal the number of provided images.",
            "Use exact image_path values from image_metadata.",
            "If uncertain, set the relevant field to uncertain and add it to uncertain_fields.",
            "Never infer brand, exact model, or license plate.",
            "Do not use background as stable identity attribute.",
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


def normalize_id_annotation(payload: dict[str, Any], vehicle_id: str) -> dict[str, Any]:
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
        "qa_status": ensure_scalar(ann.get("qa_status"), default="auto_labeled"),
    }
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
        description_zh = ann.get("description_zh", "")
        description_en = ann.get("description_en", "")
        if not description_zh or not description_en:
            qa_status = "manual_review"
            if "description" not in uncertain_fields:
                uncertain_fields.append("description")
        normalized_rows.append(
            {
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
                "confidence": normalize_confidence(ann.get("confidence")),
                "uncertain_fields": uncertain_fields,
                "qa_status": ensure_scalar(qa_status, default="auto_labeled"),
            }
        )
    return normalized_rows


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


def run_job(
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
            "model": "mock",
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
            "model": "local_color_heuristic",
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
    response = post_chat_completion(
        base_url=args.base_url,
        api_key=api_key,
        payload=request_payload,
        timeout=args.timeout,
        retries=args.retries,
    )
    text = extract_message_text(response)
    payload = parse_json_object(text)
    raw = {
        "vehicle_id": vehicle_id,
        "backend": "api",
        "model": args.model,
        "base_url": args.base_url,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selected_images": [row["image_path"] for row in rows],
        "response_text": text,
        "usage": response.get("usage"),
        "finish_reason": (response.get("choices") or [{}])[0].get("finish_reason"),
    }
    return payload, raw


def write_report(
    output_dir: Path,
    backend: str,
    model: str,
    jobs: list[tuple[str, list[dict[str, Any]]]],
    id_rows: list[dict[str, Any]],
    image_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    mock: bool,
) -> None:
    low_conf = [
        row for row in image_rows if float(row.get("confidence") or 0.0) < 0.5
    ]
    manual_review = [
        row for row in image_rows if row.get("qa_status") == "manual_review"
    ]
    usage_rows = [row.get("usage") for row in raw_rows if row.get("usage")]
    report = f"""# CVPair Annotation Smoke Test Report

Generated at: `{datetime.now(timezone.utc).isoformat()}`

- Mode: `{backend}`
- Model: `{model}`
- Vehicle IDs annotated: {len(id_rows)}
- Image annotations: {len(image_rows)}
- Low confidence image annotations: {len(low_conf)}
- Manual review image annotations: {len(manual_review)}
- API usage records with usage field: {len(usage_rows)}

## Selected Jobs

"""
    for vehicle_id, rows in jobs:
        report += f"- `{vehicle_id}`: " + ", ".join(row["image_path"] for row in rows) + "\n"
    report += """
## Quick QA Notes

- Verify that ID-level descriptions do not include background identity shortcuts.
- Verify that image-level descriptions only describe visible content.
- Treat `manual_review` and low-confidence rows as not publishable until checked.
"""
    (output_dir / "annotation_smoke_test_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    backend = "mock" if args.mock else args.backend
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
    id_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for vehicle_id, rows in jobs:
        try:
            payload, raw = run_job(vehicle_id, rows, args, api_key)
            id_rows.append(normalize_id_annotation(payload, vehicle_id))
            image_rows.extend(normalize_image_annotations(payload, rows, vehicle_id))
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

    write_jsonl(output_dir / "annotations_id.jsonl", id_rows)
    write_jsonl(output_dir / "annotations_image.jsonl", image_rows)
    write_jsonl(output_dir / "raw_responses.jsonl", raw_rows)
    write_jsonl(output_dir / "errors.jsonl", errors)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": dataset_root.as_posix(),
        "metadata_path": metadata_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "backend": backend,
        "model": "mock" if backend == "mock" else "local_color_heuristic" if backend == "heuristic" else args.model,
        "base_url": args.base_url if backend == "api" else None,
        "limit_ids": args.limit_ids,
        "images_per_id": args.images_per_id,
        "selected_vehicle_ids": [vehicle_id for vehicle_id, _ in jobs],
        "annotation_id_count": len(id_rows),
        "annotation_image_count": len(image_rows),
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
        args.mock,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
