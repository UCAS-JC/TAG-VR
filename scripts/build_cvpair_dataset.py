#!/usr/bin/env python3
"""Build a clean TAG-VR CVPair dataset copy and metadata index.

The script treats the source CVPair/CVnet directory as read-only. It preserves
the original a2g/g2a protocol in the index, while copying each unique basename
only once into the new dataset image directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Optional

from PIL import Image


PROTOCOL_SPLITS = {
    "query": "query",
    "bounding_box_train": "train",
    "bounding_box_test": "gallery",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FILENAME_RE = re.compile(
    r"^(?P<raw_vehicle_id>\d+)_"
    r"(?P<camera_id>c[01])"
    r"(?P<sequence>s\d+)_"
    r"(?P<frame>\d+)_"
    r"(?P<instance>\d+)"
    r"\.(?P<ext>jpg|jpeg|png|bmp|webp)$",
    re.IGNORECASE,
)


@dataclass
class Candidate:
    source_path: Path
    source_relative_path: str
    protocol: str
    split_dir: str
    split: str
    basename: str
    raw_vehicle_id: str
    vehicle_id: str
    camera_id: str
    view_source: str
    platform_type: str
    sequence_id: str
    frame_index: int
    instance: int
    file_format: str
    source_extension: str
    output_extension: str
    width: int
    height: int
    mode: str
    small_target: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a clean TAG-VR CVPair dataset directory."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/Volumes/AIRCAS_JC/data"),
        help="Read-only CVPair/CVnet source root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("tag_vr_dataset"),
        help="Output TAG-VR dataset root.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("copy", "symlink", "hardlink"),
        default="copy",
        help="How to materialize unique images in the new dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing copied files when materializing images.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260624,
        help="Recorded random seed for future split generation.",
    )
    parser.add_argument(
        "--source-version",
        default="local_snapshot_2026-06-24",
        help="Human-readable source snapshot/version recorded in reports.",
    )
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def image_format_to_extension(file_format: str, source_suffix: str) -> str:
    fmt = file_format.lower()
    if fmt in {"jpeg", "jpg"}:
        return "jpg"
    if fmt == "png":
        return "png"
    if fmt == "bmp":
        return "bmp"
    if fmt == "webp":
        return "webp"
    return source_suffix.lower().lstrip(".") or fmt


def detect_png_header(header: bytes) -> Optional[tuple[str, int, int, str]]:
    if not header.startswith(b"\x89PNG\r\n\x1a\n") or len(header) < 26:
        return None
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    color_type = header[25]
    mode = {
        0: "L",
        2: "RGB",
        3: "P",
        4: "LA",
        6: "RGBA",
    }.get(color_type, "unknown")
    return "png", width, height, mode


def detect_jpeg_header(path: Path) -> Optional[tuple[str, int, int, str]]:
    sof_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    with path.open("rb") as f:
        if f.read(2) != b"\xff\xd8":
            return None
        while True:
            marker_prefix = f.read(1)
            if not marker_prefix:
                return None
            while marker_prefix != b"\xff":
                marker_prefix = f.read(1)
                if not marker_prefix:
                    return None
            marker = f.read(1)
            while marker == b"\xff":
                marker = f.read(1)
            if not marker:
                return None
            marker_value = marker[0]
            if marker_value in {0xD8, 0xD9} or 0xD0 <= marker_value <= 0xD7:
                continue
            length_bytes = f.read(2)
            if len(length_bytes) != 2:
                return None
            segment_length = int.from_bytes(length_bytes, "big")
            if segment_length < 2:
                return None
            if marker_value in sof_markers:
                data = f.read(segment_length - 2)
                if len(data) < 6:
                    return None
                height = int.from_bytes(data[1:3], "big")
                width = int.from_bytes(data[3:5], "big")
                components = data[5]
                mode = "RGB" if components == 3 else "L" if components == 1 else "CMYK" if components == 4 else "unknown"
                return "jpeg", width, height, mode
            f.seek(segment_length - 2, os.SEEK_CUR)


def detect_image(path: Path) -> tuple[str, int, int, str]:
    with path.open("rb") as f:
        header = f.read(32)
    png = detect_png_header(header)
    if png:
        return png
    if header.startswith(b"\xff\xd8"):
        jpeg = detect_jpeg_header(path)
        if jpeg:
            return jpeg
    with Image.open(path) as img:
        file_format = (img.format or "unknown").lower()
        width, height = img.size
        mode = img.mode
    return file_format, width, height, mode


def is_ignored_file(path: Path) -> bool:
    name = path.name
    if name.startswith("._"):
        return True
    if name.startswith("."):
        return True
    if name.endswith(".baiduyun.uploading.cfg"):
        return True
    return False


def view_from_camera(camera_id: str) -> tuple[str, str]:
    if camera_id == "c1":
        return "uav", "uav"
    return "ground_camera", "ground_camera"


def output_name(candidate: Candidate) -> str:
    raw_id = candidate.raw_vehicle_id.zfill(4)
    frame = f"{candidate.frame_index:06d}"
    instance = f"{candidate.instance:02d}"
    return (
        f"cvpair_global_{raw_id}_{candidate.view_source}_"
        f"{candidate.sequence_id}_{frame}_{instance}.{candidate.output_extension}"
    )


def scan_source(source_root: Path) -> tuple[list[Candidate], list[dict[str, Any]], Counter]:
    candidates: list[Candidate] = []
    quality_notes: list[dict[str, Any]] = []
    ignored_counts: Counter = Counter()
    visited_files = 0

    for protocol in ("a2g", "g2a"):
        protocol_root = source_root / protocol
        for split_dir, split in PROTOCOL_SPLITS.items():
            split_root = protocol_root / split_dir
            if not split_root.exists():
                quality_notes.append(
                    {
                        "note_type": "missing_split_dir",
                        "path": str(split_root),
                        "details": "Expected protocol split directory is missing.",
                        "action": "check_source_layout",
                    }
                )
                continue

            for path in sorted(split_root.iterdir(), key=lambda p: p.name):
                if not path.is_file():
                    continue
                visited_files += 1
                if visited_files % 2000 == 0:
                    print(
                        f"[scan] visited {visited_files} source files; "
                        f"valid images {len(candidates)}",
                        file=sys.stderr,
                        flush=True,
                    )
                source_rel = path.relative_to(source_root).as_posix()
                if is_ignored_file(path):
                    ignored_counts["ignored_hidden_or_sidecar"] += 1
                    continue
                if path.suffix.lower() not in IMAGE_SUFFIXES:
                    ignored_counts["ignored_non_image_suffix"] += 1
                    quality_notes.append(
                        {
                            "note_type": "non_image_file",
                            "path": source_rel,
                            "details": f"Unsupported suffix {path.suffix!r}.",
                            "action": "excluded",
                        }
                    )
                    continue

                match = FILENAME_RE.match(path.name)
                if not match:
                    ignored_counts["invalid_filename"] += 1
                    quality_notes.append(
                        {
                            "note_type": "invalid_filename",
                            "path": source_rel,
                            "details": "Filename does not match CVPair pattern.",
                            "action": "excluded",
                        }
                    )
                    continue

                try:
                    file_format, width, height, mode = detect_image(path)
                except Exception as exc:  # noqa: BLE001 - report bad source files.
                    ignored_counts["unreadable_image"] += 1
                    quality_notes.append(
                        {
                            "note_type": "unreadable_image",
                            "path": source_rel,
                            "details": repr(exc),
                            "action": "excluded",
                        }
                    )
                    continue

                raw_vehicle_id = match.group("raw_vehicle_id")
                camera_id = match.group("camera_id").lower()
                view_source, platform_type = view_from_camera(camera_id)
                sequence_id = match.group("sequence").lower()
                frame_index = int(match.group("frame"))
                instance = int(match.group("instance"))
                output_ext = image_format_to_extension(file_format, path.suffix)
                source_ext = match.group("ext").lower()
                if output_ext != source_ext.replace("jpeg", "jpg"):
                    quality_notes.append(
                        {
                            "note_type": "extension_format_mismatch",
                            "path": source_rel,
                            "details": (
                                f"Filename extension is {source_ext}, "
                                f"but Pillow detects {file_format}."
                            ),
                            "action": f"copied_with_extension_{output_ext}",
                        }
                    )

                candidates.append(
                    Candidate(
                        source_path=path,
                        source_relative_path=source_rel,
                        protocol=protocol,
                        split_dir=split_dir,
                        split=split,
                        basename=path.name,
                        raw_vehicle_id=raw_vehicle_id,
                        vehicle_id=f"cvpair_global_{raw_vehicle_id.zfill(4)}",
                        camera_id=camera_id,
                        view_source=view_source,
                        platform_type=platform_type,
                        sequence_id=sequence_id,
                        frame_index=frame_index,
                        instance=instance,
                        file_format=file_format,
                        source_extension=source_ext,
                        output_extension=output_ext,
                        width=width,
                        height=height,
                        mode=mode,
                        small_target=min(width, height) < 64,
                    )
                )

    return candidates, quality_notes, ignored_counts


def materialize_image(src: Path, dst: Path, link_mode: str, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if overwrite:
            dst.unlink()
        else:
            return
    if link_mode == "copy":
        shutil.copy2(src, dst)
    elif link_mode == "symlink":
        os.symlink(src, dst)
    elif link_mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError(f"Unsupported link mode: {link_mode}")


def build_indexes(
    candidates: list[Candidate],
    output_root: Path,
    source_root: Path,
    link_mode: str,
    overwrite: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    basename_counts = Counter(c.basename for c in candidates)
    by_basename: dict[str, Candidate] = {}
    for candidate in candidates:
        by_basename.setdefault(candidate.basename, candidate)

    image_path_by_basename: dict[str, str] = {}
    for basename, candidate in sorted(by_basename.items()):
        rel_image_path = f"images/cvpair/{output_name(candidate)}"
        image_path_by_basename[basename] = rel_image_path
        materialize_image(
            candidate.source_path,
            output_root / rel_image_path,
            link_mode=link_mode,
            overwrite=overwrite,
        )

    records: list[dict[str, Any]] = []
    for candidate in candidates:
        target_size = {
            "width_px": candidate.width,
            "height_px": candidate.height,
            "short_side_px": min(candidate.width, candidate.height),
        }
        rel_image_path = image_path_by_basename[candidate.basename]
        records.append(
            {
                "dataset": "tag_vr",
                "source_dataset": "cvpair",
                "source_domain": "real",
                "source_task": "aerial_ground_vehicle_reid",
                "protocol": candidate.protocol,
                "split": candidate.split,
                "source_split_dir": candidate.split_dir,
                "vehicle_id": candidate.vehicle_id,
                "raw_vehicle_id": candidate.raw_vehicle_id,
                "identity_scope": "global",
                "identity_confidence": "global_id_verified",
                "association_method": "cvpair_vehicle_id",
                "image_path": rel_image_path,
                "source_relative_path": candidate.source_relative_path,
                "original_path": str(candidate.source_path),
                "camera_id": candidate.camera_id,
                "view_source": candidate.view_source,
                "platform_type": candidate.platform_type,
                "sequence": candidate.sequence_id,
                "sequence_id": candidate.sequence_id,
                "frame": candidate.frame_index,
                "frame_index": candidate.frame_index,
                "instance": candidate.instance,
                "file_format": candidate.file_format,
                "source_extension": candidate.source_extension,
                "file_format_checked": True,
                "duplicate_basename": basename_counts[candidate.basename] > 1,
                "target_size": target_size,
                "small_target": candidate.small_target,
                "qa_status": "raw",
            }
        )

    metadata_by_image: dict[str, dict[str, Any]] = {}
    memberships_by_image: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        memberships_by_image[record["image_path"]].append(
            {
                "protocol": record["protocol"],
                "split": record["split"],
                "source_relative_path": record["source_relative_path"],
            }
        )
        if record["image_path"] in metadata_by_image:
            continue
        metadata_by_image[record["image_path"]] = {
            "dataset": "tag_vr",
            "source_dataset": "cvpair",
            "source_domain": "real",
            "source_task": "aerial_ground_vehicle_reid",
            "source_split": record["split"],
            "vehicle_id": record["vehicle_id"],
            "raw_vehicle_id": record["raw_vehicle_id"],
            "track_id": None,
            "global_id": record["raw_vehicle_id"],
            "identity_scope": "global",
            "identity_confidence": "global_id_verified",
            "association_method": "cvpair_vehicle_id",
            "scene_id": "cvpair",
            "sequence_id": record["sequence_id"],
            "timestamp": None,
            "frame_index": record["frame_index"],
            "platform_type": record["platform_type"],
            "platform_id": None,
            "sensor_id": record["camera_id"],
            "camera_id": record["camera_id"],
            "view_source": record["view_source"],
            "image_path": record["image_path"],
            "full_frame_path": None,
            "bbox_2d": None,
            "bbox_3d": None,
            "pose": None,
            "category": "vehicle",
            "target_size": record["target_size"],
            "small_target": record["small_target"],
            "conversion_status": "converted",
            "qa_status": "raw",
            "original_paths": [record["original_path"]],
            "protocol_memberships": [],
        }

    for image_path, memberships in memberships_by_image.items():
        row = metadata_by_image[image_path]
        row["protocol_memberships"] = sorted(
            memberships, key=lambda m: (m["protocol"], m["split"], m["source_relative_path"])
        )
        row["original_paths"] = sorted(
            {
                record["original_path"]
                for record in records
                if record["image_path"] == image_path
            }
        )

    metadata = [metadata_by_image[key] for key in sorted(metadata_by_image)]

    links_by_vehicle: dict[str, dict[str, Any]] = {}
    for row in metadata:
        link = links_by_vehicle.setdefault(
            row["vehicle_id"],
            {
                "vehicle_id": row["vehicle_id"],
                "source_dataset": "cvpair",
                "identity_scope": "global",
                "identity_confidence": "global_id_verified",
                "association_method": "cvpair_vehicle_id",
                "association_evidence": [
                    "cvpair_same_vehicle_id",
                    "official_a2g_g2a_protocol",
                ],
                "uav_samples": [],
                "ground_samples": [],
                "reviewer": None,
                "qa_status": "raw",
            },
        )
        if row["view_source"] == "uav":
            link["uav_samples"].append(row["image_path"])
        else:
            link["ground_samples"].append(row["image_path"])

    identity_links = []
    for link in links_by_vehicle.values():
        link["uav_samples"] = sorted(set(link["uav_samples"]))
        link["ground_samples"] = sorted(set(link["ground_samples"]))
        identity_links.append(link)
    identity_links.sort(key=lambda row: row["vehicle_id"])

    return records, metadata, identity_links, image_path_by_basename


def write_quality_notes(path: Path, notes: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["note_type", "path", "details", "action"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for note in notes:
            writer.writerow({field: note.get(field, "") for field in fieldnames})


def build_splits(records: list[dict[str, Any]], metadata: list[dict[str, Any]]) -> dict[str, Any]:
    train_ids = sorted(
        {row["vehicle_id"] for row in records if row["split"] == "train"}
    )
    test_ids = sorted(
        {row["vehicle_id"] for row in records if row["split"] in {"query", "gallery"}}
    )
    train_images = sorted(
        {row["image_path"] for row in metadata if row["vehicle_id"] in train_ids}
    )
    test_images = sorted(
        {row["image_path"] for row in metadata if row["vehicle_id"] in test_ids}
    )
    return {
        "official_cvpair_protocol": {
            "train_vehicle_ids": train_ids,
            "val_vehicle_ids": [],
            "test_vehicle_ids": test_ids,
            "official_val_available": False,
            "note": "CVPair/CVnet provides train and query/gallery test IDs; no official validation split was found.",
        },
        "train": {"vehicle_ids": train_ids, "image_paths": train_images},
        "val": {"vehicle_ids": [], "image_paths": []},
        "test": {"vehicle_ids": test_ids, "image_paths": test_images},
    }


def write_split_files(output_root: Path, split_payload: dict[str, Any]) -> None:
    write_json(output_root / "metadata" / "split.json", split_payload)
    for split_name in ("train", "val", "test"):
        write_json(output_root / "splits" / f"{split_name}.json", split_payload[split_name])


def write_benchmarks(output_root: Path, records: list[dict[str, Any]]) -> None:
    by_protocol_split: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_protocol_split[(row["protocol"], row["split"])].append(row)

    for key in by_protocol_split:
        by_protocol_split[key].sort(key=lambda r: r["image_path"])

    for protocol, task_name in (("a2g", "visual_a2g"), ("g2a", "visual_g2a")):
        gallery = by_protocol_split[(protocol, "gallery")]
        gallery_by_vehicle: defaultdict[str, list[str]] = defaultdict(list)
        for row in gallery:
            gallery_by_vehicle[row["vehicle_id"]].append(row["image_path"])

        rows = []
        for query in by_protocol_split[(protocol, "query")]:
            rows.append(
                {
                    "dataset": "tag_vr",
                    "task": task_name,
                    "protocol": protocol,
                    "query_image": query["image_path"],
                    "query_vehicle_id": query["vehicle_id"],
                    "query_camera_id": query["camera_id"],
                    "query_view_source": query["view_source"],
                    "gallery_split": "gallery",
                    "gallery_image_count": len(gallery),
                    "positive_gallery_images": sorted(
                        gallery_by_vehicle.get(query["vehicle_id"], [])
                    ),
                    "metric": ["mAP", "Rank-1", "Rank-5", "Rank-10"],
                }
            )
        write_jsonl(output_root / "benchmarks" / f"{task_name}.jsonl", rows)

    empty_tasks = [
        "text_to_ground",
        "text_to_uav",
        "image_to_text",
        "text_guided_cross_view",
        "attribute_retrieval",
        "cross_source_generalization",
    ]
    for task in empty_tasks:
        (output_root / "benchmarks").mkdir(parents=True, exist_ok=True)
        (output_root / "benchmarks" / f"{task}.jsonl").write_text("", encoding="utf-8")


def write_configs(output_root: Path, source_root: Path, seed: int) -> None:
    configs_dir = output_root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    (configs_dir / "conversion_config.yaml").write_text(
        f"""# Generated by scripts/build_cvpair_dataset.py
source_dataset: cvpair
source_root: {source_root.as_posix()}
output_root: {output_root.as_posix()}
random_seed: {seed}
protocols:
  - a2g
  - g2a
split_mapping:
  query: query
  bounding_box_train: train
  bounding_box_test: gallery
view_mapping:
  c0: ground_camera
  c1: uav
identity_scope: global
identity_confidence: global_id_verified
copy_policy: unique_basename
ignored_files:
  - "._*"
  - ".*"
  - "*.baiduyun.uploading.cfg"
""",
        encoding="utf-8",
    )
    (configs_dir / "schema_mapping.yaml").write_text(
        """# Generated by scripts/build_cvpair_dataset.py
raw_filename_pattern: "{raw_vehicle_id}_{camera_id}{sequence}_{frame}_{instance}.{ext}"
fields:
  raw_vehicle_id: raw filename id, zero-padded to four digits for vehicle_id
  vehicle_id: "cvpair_global_{raw_vehicle_id:04d}"
  camera_id: c0 or c1
  view_source:
    c0: ground_camera
    c1: uav
  sequence_id: filename sequence token, e.g. s2
  frame_index: filename frame token
  instance: filename instance token
  category: vehicle
  bbox_2d: null, because source files are already vehicle crops
""",
        encoding="utf-8",
    )
    (configs_dir / "label_prompt.yaml").write_text(
        """# Prompt constraints used by scripts/annotate_cvpair_text.py
language: zh_en
annotation_levels:
  - id_level_stable_description
  - image_level_visible_attributes
required_fields:
  - description_zh
  - description_en
  - color
  - vehicle_type
  - orientation
  - visible_parts
  - occlusion
  - scene_context
  - confidence
  - qa_status
forbidden:
  - Do not infer brand.
  - Do not infer exact model.
  - Do not use license plate numbers as identity features.
  - Do not describe invisible details.
  - Do not use parking slots, road texture, poles, or neighboring vehicles as stable identity attributes.
uncertainty:
  - Use uncertain for small targets, occlusion, low light, or ambiguous color/type.
""",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_readme(output_root: Path, stats: dict[str, Any]) -> None:
    readme = f"""# TAG-VR CVPair Clean Dataset

Generated at: `{stats["generated_at"]}`

This directory is a clean, derived CVPair/CVnet subset for TAG-VR. The original
source under `{stats["source_root"]}` was treated as read-only. Protocol-level
records are preserved in `cvpair_index/cvpair_clean_index.jsonl`, while duplicate
training basenames from `a2g` and `g2a` are materialized only once under
`images/cvpair/`.

## Key Counts

| Item | Count |
| --- | ---: |
| Protocol image records | {stats["protocol_records"]} |
| Unique image basenames | {stats["unique_basenames"]} |
| Vehicle IDs | {stats["vehicle_ids"]} |
| Train IDs | {stats["train_ids"]} |
| Test IDs | {stats["test_ids"]} |
| Duplicate basename records | {stats["duplicate_record_count"]} |
| Extension/format mismatches | {stats["format_mismatch_count"]} |

## Main Files

- `cvpair_index/cvpair_clean_index.jsonl`: protocol-preserving CVPair index.
- `metadata/metadata.jsonl`: one row per unique materialized crop.
- `metadata/identity_links.jsonl`: CVPair ID-level aerial-ground associations.
- `metadata/split.json`: official train/test ID split; no official val split.
- `benchmarks/visual_a2g.jsonl` and `benchmarks/visual_g2a.jsonl`: visual retrieval task indexes.
- `configs/label_prompt.yaml`: annotation constraints for VLM labeling.
- `qa/qa_report.md`: generation statistics, known issues, and risk notes.

## Usage Boundary

Do not treat empty text benchmark files as complete until
`scripts/annotate_cvpair_text.py` has produced reviewed annotations. CVPair is
the real Re-ID core of TAG-VR; extension datasets should be converted and
validated separately before being merged.
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")


def write_qa_report(
    output_root: Path,
    stats: dict[str, Any],
    split_counts: Counter,
    camera_counts: Counter,
    format_counts: Counter,
    ignored_counts: Counter,
) -> None:
    split_lines = "\n".join(
        f"| `{protocol}` | `{split}` | {count} |"
        for (protocol, split), count in sorted(split_counts.items())
    )
    camera_lines = "\n".join(
        f"| `{camera}` | {count} |" for camera, count in sorted(camera_counts.items())
    )
    format_lines = "\n".join(
        f"| `{fmt}` | {count} |" for fmt, count in sorted(format_counts.items())
    )
    ignored_lines = "\n".join(
        f"| `{key}` | {value} |" for key, value in sorted(ignored_counts.items())
    )
    if not ignored_lines:
        ignored_lines = "| none | 0 |"

    report = f"""# CVPair QA Report

Generated at: `{stats["generated_at"]}`

## Source

- Source dataset: CVPair / CVnet
- Source root: `{stats["source_root"]}`
- Source version/snapshot: `{stats["source_version"]}`
- License status: `to_be_confirmed_before_public_release`
- Original task: aerial-ground vehicle Re-ID

## Scale

| Metric | Count |
| --- | ---: |
| Protocol image records | {stats["protocol_records"]} |
| Unique materialized images | {stats["unique_basenames"]} |
| Vehicle IDs | {stats["vehicle_ids"]} |
| Train IDs | {stats["train_ids"]} |
| Test IDs | {stats["test_ids"]} |
| Train/test ID overlap | {stats["train_test_overlap"]} |
| Duplicate basename records | {stats["duplicate_record_count"]} |
| Small-target images | {stats["small_target_count"]} |
| Extension/format mismatches | {stats["format_mismatch_count"]} |

## Protocol Counts

| Protocol | Split | Records |
| --- | --- | ---: |
{split_lines}

## Camera Counts

| Camera | Records |
| --- | ---: |
{camera_lines}

## Detected File Formats

| Format | Records |
| --- | ---: |
{format_lines}

## Ignored Or Excluded Source Files

| Type | Count |
| --- | ---: |
{ignored_lines}

## Identity Reliability

All CVPair rows use `identity_confidence="global_id_verified"` because the
official CVPair/CVnet protocol supplies same-ID aerial and ground crops. No
extension samples were mixed into this output.

## Text Coverage

ID-level text coverage and image-level text coverage are not complete in this
directory until the VLM annotation script is run and manually reviewed. The
smoke-test annotation output should be stored under
`annotations/cvpair_smoke_test/`.

## Known Risks

- UAV-side targets can be small or partially occluded; use `uncertain` where
  color, type, or visible parts are ambiguous.
- Ground-side crops can include neighboring vehicles, poles, motorcycles, and
  garage background clutter.
- Do not use parking slots, road texture, pillars, or neighboring-vehicle
  layouts as stable identity descriptions.
- Some files have `.jpg` names but PNG bytes; this was recorded in
  `cvpair_index/cvpair_quality_notes.csv` and materialized with an extension
  matching the detected file format.
"""
    (output_root / "qa").mkdir(parents=True, exist_ok=True)
    (output_root / "qa" / "qa_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if not source_root.exists():
        raise SystemExit(f"Source root does not exist: {source_root}")

    for dirname in [
        "cvpair_index",
        "images/cvpair",
        "metadata",
        "splits",
        "annotations",
        "benchmarks",
        "configs",
        "qa",
        "source_audit",
    ]:
        (output_root / dirname).mkdir(parents=True, exist_ok=True)

    candidates, quality_notes, ignored_counts = scan_source(source_root)
    if not candidates:
        raise SystemExit(f"No valid CVPair images found under {source_root}")

    records, metadata, identity_links, _ = build_indexes(
        candidates,
        output_root=output_root,
        source_root=source_root,
        link_mode=args.link_mode,
        overwrite=args.overwrite,
    )

    write_jsonl(output_root / "cvpair_index" / "cvpair_clean_index.jsonl", records)
    write_jsonl(output_root / "metadata" / "metadata.jsonl", metadata)
    write_jsonl(output_root / "metadata" / "identity_links.jsonl", identity_links)

    split_payload = build_splits(records, metadata)
    write_split_files(output_root, split_payload)
    write_benchmarks(output_root, records)
    write_configs(output_root, source_root, args.seed)

    train_ids = set(split_payload["official_cvpair_protocol"]["train_vehicle_ids"])
    test_ids = set(split_payload["official_cvpair_protocol"]["test_vehicle_ids"])
    basename_counts = Counter(row["source_relative_path"].split("/")[-1] for row in records)
    duplicate_record_count = sum(1 for row in records if row["duplicate_basename"])
    format_mismatch_count = sum(
        1 for note in quality_notes if note["note_type"] == "extension_format_mismatch"
    )
    split_counts = Counter((row["protocol"], row["split"]) for row in records)
    camera_counts = Counter(row["camera_id"] for row in records)
    format_counts = Counter(row["file_format"] for row in records)
    stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_root": source_root.as_posix(),
        "source_version": args.source_version,
        "output_root": output_root.as_posix(),
        "protocol_records": len(records),
        "unique_basenames": len(basename_counts),
        "vehicle_ids": len({row["vehicle_id"] for row in records}),
        "train_ids": len(train_ids),
        "test_ids": len(test_ids),
        "train_test_overlap": len(train_ids & test_ids),
        "duplicate_record_count": duplicate_record_count,
        "small_target_count": sum(1 for row in metadata if row["small_target"]),
        "format_mismatch_count": format_mismatch_count,
        "link_mode": args.link_mode,
        "cvpair_clean_index_sha256": sha256_file(
            output_root / "cvpair_index" / "cvpair_clean_index.jsonl"
        ),
    }

    write_quality_notes(output_root / "cvpair_index" / "cvpair_quality_notes.csv", quality_notes)
    write_json(output_root / "metadata" / "build_summary.json", stats)
    write_readme(output_root, stats)
    write_qa_report(
        output_root=output_root,
        stats=stats,
        split_counts=split_counts,
        camera_counts=camera_counts,
        format_counts=format_counts,
        ignored_counts=ignored_counts,
    )

    (output_root / "source_audit" / "source_audit.md").write_text(
        f"""# Source Audit

## CVPair / CVnet

- Source root: `{source_root.as_posix()}`
- Snapshot label: `{args.source_version}`
- Original task: aerial-ground vehicle Re-ID
- Public release/license: to be confirmed before public redistribution
- Processing status: clean index and unique crop copy generated
- Identity status: official same-ID labels treated as `global_id_verified`

No extension datasets were processed in this run.
""",
        encoding="utf-8",
    )
    (output_root / "source_audit" / "source_schema_notes.md").write_text(
        """# Source Schema Notes

CVPair filenames are parsed as:

```text
{raw_vehicle_id}_{camera_id}{sequence_id}_{frame_index}_{instance}.{ext}
```

`c0` is mapped to `ground_camera`, and `c1` is mapped to `uav`.
The source files are already vehicle crops, so `bbox_2d` is `null` and
`full_frame_path` is unavailable for this source snapshot.
""",
        encoding="utf-8",
    )

    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
