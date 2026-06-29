from __future__ import annotations

import copy
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "annotate_cvpair_text.py"
SPEC = importlib.util.spec_from_file_location("annotate_cvpair_text", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
annotator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(annotator)


ID_ZH = (
    "一辆银灰色紧凑型乘用车，车身轮廓较低且比例紧凑，前挡风玻璃和深色车窗清晰可见；"
    "跨视角稳定线索为银灰车身、低矮轮廓和紧凑比例，具体车型仍需人工复核。"
)
ID_ZH_ALT = (
    "该车呈银灰色紧凑乘用车外观，整体车身较低，挡风玻璃和深色窗区在多视角中较稳定；"
    "可确认银灰车身和紧凑比例，细分车型与局部结构仍保留不确定性。"
)
ID_EN = (
    "A silver-gray compact passenger vehicle with a low body profile, compact proportions, "
    "visible windshield, and dark window areas. The stable cross-view cues are the silver body "
    "color, low outline, and compact shape, while the exact type still needs review."
)
ID_EN_ALT = (
    "This vehicle appears as a silver-gray compact passenger car with a relatively low outline, "
    "dark windows, and visible windshield areas. Across views, the reliable cues are the silver "
    "body and compact proportions, while fine type details remain uncertain."
)

GROUND_ZH = (
    "地面近景中主车呈银灰色紧凑轮廓，可见前挡风玻璃、深色侧窗和低矮车身；"
    "邻近车辆只作为背景干扰，车型细节因角度仍需复核。"
)
GROUND_ZH_ALT = (
    "当前地面视角显示银灰色主车，车身比例紧凑且较低，挡风玻璃和侧窗区域较清楚；"
    "周围车辆属于干扰背景，细分车型和局部结构仍不确定。"
)
GROUND_EN = (
    "In this ground-view crop, the main vehicle appears silver-gray with a compact low outline, "
    "visible windshield, dark side windows, and side body. Nearby vehicles are only background "
    "clutter, and the fine type remains uncertain because of the angle."
)
GROUND_EN_ALT = (
    "From the ground perspective, the target vehicle shows a silver-gray body, compact low "
    "proportions, visible windshield, and dark side windows. Neighboring cars should be treated "
    "as background distractors, and fine structural details remain uncertain."
)

UAV_ZH = (
    "空中俯视图中主车呈银灰色紧凑形态，可见车顶、前挡风玻璃和车身边界；"
    "周围停车环境仅作为干扰背景，细节结构因俯视距离仍需复核。"
)
UAV_ZH_ALT = (
    "该空中视角下目标车辆为银灰色紧凑外观，车顶区域、挡风玻璃位置和车身外缘可见；"
    "停车场纹理和邻车不作为身份线索，精细部件仍存在不确定性。"
)
UAV_EN = (
    "In the aerial top view, the main vehicle shows a silver-gray compact shape with visible "
    "roof, windshield area, and body boundary. The parking surroundings are only contextual "
    "clutter, and fine structural details remain uncertain from the overhead distance."
)
UAV_EN_ALT = (
    "From the overhead view, the target vehicle keeps a silver-gray compact profile with visible "
    "roof area, windshield position, and outer body contour. Parking textures and neighboring "
    "cars are background clutter, while small structural details remain uncertain."
)


def make_source_rows(source_split: str = "train") -> list[dict]:
    rows = []
    specs = [
        ("images/cvpair/v001_ground_000010.jpg", "ground_camera", "c0", 96),
        ("images/cvpair/v001_ground_000030.jpg", "ground_camera", "c0", 88),
        ("images/cvpair/v001_uav_000010.jpg", "uav", "c1", 72),
        ("images/cvpair/v001_uav_000020.jpg", "uav", "c1", 64),
    ]
    for image_path, view_source, camera_id, short_side in specs:
        rows.append(
            {
                "image_path": image_path,
                "vehicle_id": "cvpair_global_0001",
                "source_dataset": "cvpair",
                "camera_id": camera_id,
                "view_source": view_source,
                "platform_type": "ground" if view_source == "ground_camera" else "uav",
                "target_size": {"width_px": short_side + 20, "height_px": short_side, "short_side_px": short_side},
                "small_target": False,
                "source_split": source_split,
                "protocol_memberships": [{"protocol": "tag_vr", "split": source_split}],
            }
        )
    return rows


def make_payload(rows: list[dict]) -> dict:
    ground_path = next(row["image_path"] for row in rows if row["view_source"] == "ground_camera")
    uav_path = next(row["image_path"] for row in rows if row["view_source"] == "uav")

    def perception_attributes(row: dict) -> list[dict]:
        return [
            {
                "name": "color",
                "value": "silver-gray",
                "confidence": 0.92,
                "visibility": "visible",
                "visual_evidence": "silver-gray painted body is visible on the crop",
                "source_images": [row["image_path"]],
            },
            {
                "name": "body_profile",
                "value": "compact_low",
                "confidence": 0.86,
                "visibility": "visible",
                "visual_evidence": "compact proportions and low body outline are visible",
                "source_images": [row["image_path"]],
            },
            {
                "name": "exact_vehicle_type",
                "value": "uncertain",
                "confidence": 0.45,
                "visibility": "partially_visible",
                "visual_evidence": "crop angle and distance do not support a fine-grained type",
                "source_images": [row["image_path"]],
            },
        ]

    annotations_image = []
    semantic_images = []
    perception_images = []
    for row in rows:
        is_uav = row["view_source"] == "uav"
        annotations_image.append(
            {
                "image_path": row["image_path"],
                "vehicle_id": "cvpair_global_0001",
                "source_dataset": "cvpair",
                "camera_id": row["camera_id"],
                "view_source": row["view_source"],
                "platform_type": row["platform_type"],
                "color": "silver-gray",
                "vehicle_type": "compact passenger vehicle",
                "orientation": "top" if is_uav else "front_left",
                "visible_parts": ["roof", "windshield", "body_boundary"] if is_uav else ["windshield", "side_windows", "side_body"],
                "occlusion": "none_obvious",
                "scene_context": "parking context with neighboring vehicles as distractors",
                "weather": None,
                "illumination": "daylight",
                "background_distractors": {
                    "neighbor_vehicle_count": 2,
                    "occluders": [],
                    "complexity": "moderate",
                },
                "description_zh": UAV_ZH if is_uav else GROUND_ZH,
                "description_en": UAV_EN if is_uav else GROUND_EN,
                "description_zh_variants": [UAV_ZH_ALT if is_uav else GROUND_ZH_ALT],
                "description_en_variants": [UAV_EN_ALT if is_uav else GROUND_EN_ALT],
                "confidence": 0.88,
                "uncertain_fields": ["exact_vehicle_type"],
                "qa_status": "auto_labeled",
            }
        )
        semantic_images.append(
            {
                "image_path": row["image_path"],
                "category": "match",
                "match_score": 0.91,
                "field_issues": [],
                "corrections": [],
                "omitted_attributes": [],
            }
        )
        perception_images.append(
            {
                "image_path": row["image_path"],
                "view_source": row["view_source"],
                "attributes": perception_attributes(row),
            }
        )

    return {
        "perception": {"images": perception_images},
        "cross_view_consensus": {
            "stable": [
                {
                    "attribute": "color",
                    "value": "silver-gray",
                    "confidence": 0.91,
                    "visual_evidence": "silver-gray body appears in both ground and UAV crops",
                    "source_images": [ground_path, uav_path],
                },
                {
                    "attribute": "body_profile",
                    "value": "compact_low",
                    "confidence": 0.84,
                    "visual_evidence": "compact low proportions are visible from both views",
                    "source_images": [ground_path, uav_path],
                },
            ],
            "view_specific": [
                {
                    "attribute": "roof_visibility",
                    "value": "clearer_in_uav",
                    "source_images": [uav_path],
                }
            ],
            "conflict": [],
            "uncertain": [
                {
                    "attribute": "exact_vehicle_type",
                    "reason": "fine type is not reliable from crop angle and distance",
                    "source_images": [ground_path, uav_path],
                }
            ],
        },
        "annotation_id": {
            "description_zh": ID_ZH,
            "description_en": ID_EN,
            "description_zh_variants": [ID_ZH_ALT],
            "description_en_variants": [ID_EN_ALT],
            "color": "silver-gray",
            "vehicle_type": "compact passenger vehicle",
            "body_profile": "compact_low",
            "roof_features": ["visible roof area in aerial view"],
            "window_features": ["visible windshield", "dark side windows"],
            "cargo_or_rear_structure": "uncertain",
            "special_marks": [],
            "stable_attributes": ["silver-gray body", "compact low proportions", "dark window areas"],
            "uncertain_attributes": ["exact_vehicle_type", "fine structural details"],
            "qa_status": "auto_labeled",
        },
        "annotations_image": annotations_image,
        "semantic_audit": {
            "id": {
                "category": "match",
                "match_score": 0.93,
                "field_issues": [],
                "corrections": [],
                "omitted_attributes": [],
            },
            "images": semantic_images,
        },
        "qa_notes": [],
    }


def make_v4_perception(rows: list[dict], uav_color: str = "gray", ground_color: str = "gray") -> dict:
    images = []
    for row in rows:
        color = uav_color if row["view_source"] == "uav" else ground_color
        parts = (
            ["roof", "windshield", "body_boundary"]
            if row["view_source"] == "uav"
            else ["hood", "windshield", "side_body"]
        )
        attributes = [
            {
                "attribute": "color",
                "value": color,
                "confidence": 0.92,
                "visibility": "visible",
                "visual_evidence": f"{color} body paint is visible",
                "source_images": [row["image_path"]],
            }
        ]
        attributes.extend(
            {
                "attribute": "visible_part",
                "value": part,
                "confidence": 0.9,
                "visibility": "visible",
                "visual_evidence": f"{part} is visible in the current crop",
                "source_images": [row["image_path"]],
            }
            for part in parts
        )
        images.append(
            {
                "image_path": row["image_path"],
                "attributes": attributes,
                "quality_notes": [],
            }
        )
    return annotator.normalize_perception_payload({"perception": {"images": images}}, rows)


def make_v4_caption_payload(rows: list[dict]) -> dict:
    perception = make_v4_perception(rows)
    consensus, perception = annotator.finalize_cross_view_consensus(perception, {})
    payload = {
        "perception": perception,
        "cross_view_consensus": consensus,
        "evidence_confidence_threshold": 0.75,
        "new_observations": [],
    }
    fact_tables = annotator.build_authorized_fact_tables(payload, rows, 0.75)

    def positive_claims(facts: list[dict]) -> list[dict]:
        return [
            {
                "claim": f"{fact['attribute']} is {fact['value']}",
                "claim_type": "positive",
                "attribute": fact["attribute"],
                "value": fact["value"],
                "fact_ids": [fact["fact_id"]],
                "reason": "",
            }
            for fact in facts
        ]

    id_claims = positive_claims(fact_tables["id"]["facts"])
    payload["annotation_id"] = {
        "description_zh": ID_ZH,
        "description_en": ID_EN,
        "description_zh_variants": [ID_ZH_ALT],
        "description_en_variants": [ID_EN_ALT],
        "caption_claims": {
            "canonical": id_claims,
            "natural_paraphrase_1": list(id_claims),
        },
        "color": "gray",
        "vehicle_type": "uncertain",
        "body_profile": "uncertain",
        "roof_features": [],
        "window_features": [],
        "cargo_or_rear_structure": "uncertain",
        "special_marks": [],
        "stable_attributes": ["color"],
        "uncertain_attributes": [],
        "qa_status": "auto_labeled",
    }
    annotations_image = []
    for row in rows:
        is_uav = row["view_source"] == "uav"
        facts = fact_tables["images"][row["image_path"]]["facts"]
        claims = positive_claims(facts)
        annotations_image.append(
            {
                "image_path": row["image_path"],
                "vehicle_id": row["vehicle_id"],
                "camera_id": row["camera_id"],
                "view_source": row["view_source"],
                "platform_type": row["platform_type"],
                "description_zh": UAV_ZH if is_uav else GROUND_ZH,
                "description_en": UAV_EN if is_uav else GROUND_EN,
                "description_zh_variants": [UAV_ZH_ALT if is_uav else GROUND_ZH_ALT],
                "description_en_variants": [UAV_EN_ALT if is_uav else GROUND_EN_ALT],
                "caption_claims": {
                    "canonical": claims,
                    "natural_paraphrase_1": list(claims),
                },
                "color": "gray",
                "vehicle_type": "uncertain",
                "orientation": "top" if is_uav else "front",
                "visible_parts": [fact["value"] for fact in facts if fact["attribute"] == "visible_part"],
                "occlusion": "none_obvious",
                "scene_context": "parking context",
                "confidence": 0.9,
                "uncertain_fields": [],
                "qa_status": "auto_labeled",
            }
        )
    payload["annotations_image"] = annotations_image
    return payload


def make_supported_claim_audit(rows: list[dict], failed_first_claim: bool = False) -> dict:
    id_items = []
    image_items = []
    for variant in annotator.caption_variant_names(2):
        id_items.append(
            {
                "variant": variant,
                "all_caption_claims_covered": True,
                "claims": [
                    {
                        "claim": "the vehicle body is gray",
                        "attribute": "color",
                        "value": "gray",
                        "status": "supported",
                        "source_images": [rows[0]["image_path"], rows[2]["image_path"]],
                        "correction": "",
                    }
                ],
            }
        )
        for row in rows:
            image_items.append(
                {
                    "image_path": row["image_path"],
                    "variant": variant,
                    "all_caption_claims_covered": True,
                    "claims": [
                        {
                            "claim": "the vehicle body is gray",
                            "attribute": "color",
                            "value": "gray",
                            "status": "supported",
                            "source_images": [row["image_path"]],
                            "correction": "",
                        }
                    ],
                }
            )
    if failed_first_claim:
        image_items[0]["claims"][0]["status"] = "not_visible"
        image_items[0]["claims"][0]["correction"] = "remove the unsupported claim"
    return {"id": id_items, "images": image_items}


def run_local_qa_pipeline(source_split: str = "train") -> tuple[dict, list[dict], dict, list[dict], list[dict]]:
    rows = make_source_rows(source_split)
    payload = make_payload(rows)
    vehicle_id = "cvpair_global_0001"
    id_row = annotator.normalize_id_annotation(payload, vehicle_id, caption_variants=2)
    image_rows = annotator.normalize_image_annotations(payload, rows, vehicle_id, caption_variants=2)
    semantic_qa_rows = annotator.build_semantic_qa_rows(payload, vehicle_id, image_rows, 0.75)
    annotator.apply_semantic_qa(id_row, image_rows, semantic_qa_rows)
    evidence_row = annotator.build_annotation_evidence_row(payload, vehicle_id, rows, "unit", "cot_audit")
    annotator.apply_evidence_qa(id_row, image_rows, evidence_row)
    annotator.apply_human_review_policy(id_row, image_rows, rows, enforce_eval_review=True)
    caption_qa_rows = annotator.apply_caption_length_qa([id_row], image_rows, expected_caption_variants=2)
    return id_row, image_rows, evidence_row, semantic_qa_rows, caption_qa_rows


class AnnotateCvpairTextV3Test(unittest.TestCase):
    def test_balanced_representative_selection_prefers_two_ground_and_two_uav(self) -> None:
        selected = annotator.select_representative_images(make_source_rows("train"), images_per_id=4)
        views = [row["view_source"] for row in selected]
        self.assertEqual(views.count("ground_camera"), 2)
        self.assertEqual(views.count("uav"), 2)

    def test_valid_v3_train_payload_passes_quality_gates(self) -> None:
        id_row, image_rows, evidence_row, semantic_qa_rows, caption_qa_rows = run_local_qa_pipeline("train")

        self.assertEqual(evidence_row["evidence_qa"]["status"], "pass", evidence_row["evidence_qa"])
        self.assertTrue(all(row["status"] == "pass" for row in semantic_qa_rows), semantic_qa_rows)
        self.assertTrue(all(row["status"] == "pass" for row in caption_qa_rows), caption_qa_rows)
        self.assertEqual(id_row["qa_status"], "auto_labeled")
        self.assertFalse(id_row["review_required"])
        self.assertEqual(id_row["review_reasons"], [])
        for image_row in image_rows:
            self.assertEqual(image_row["qa_status"], "auto_labeled", image_row)
            self.assertFalse(image_row["review_required"])
            self.assertEqual(image_row["review_reasons"], [])

    def test_eval_samples_require_human_review_even_when_auto_qa_passes(self) -> None:
        id_row, image_rows, evidence_row, semantic_qa_rows, caption_qa_rows = run_local_qa_pipeline("query")

        self.assertEqual(evidence_row["evidence_qa"]["status"], "pass", evidence_row["evidence_qa"])
        self.assertTrue(all(row["status"] == "pass" for row in semantic_qa_rows), semantic_qa_rows)
        self.assertTrue(all(row["status"] == "pass" for row in caption_qa_rows), caption_qa_rows)
        self.assertEqual(id_row["qa_status"], "manual_review")
        self.assertTrue(id_row["review_required"])
        self.assertIn("evaluation_identity", id_row["review_reasons"])
        for image_row in image_rows:
            self.assertEqual(image_row["qa_status"], "manual_review", image_row)
            self.assertTrue(image_row["review_required"])
            self.assertIn("evaluation_sample", image_row["review_reasons"])

    def test_local_caption_density_repair_converts_short_payload_to_length_pass(self) -> None:
        rows = make_source_rows("train")
        payload = make_payload(rows)
        payload["annotation_id"]["description_zh"] = "灰色轿车，外观稳定。"
        payload["annotation_id"]["description_en"] = "A gray sedan with stable appearance."
        payload["annotation_id"]["description_zh_variants"] = []
        payload["annotation_id"]["description_en_variants"] = []
        for item in payload["annotations_image"]:
            item["description_zh"] = "灰色轿车，局部遮挡。"
            item["description_en"] = "A gray sedan with partial occlusion."
            item["description_zh_variants"] = []
            item["description_en_variants"] = []

        before = annotator.preview_caption_length_qa(payload, rows, "cvpair_global_0001", caption_variants=2)
        self.assertTrue(any(row["status"] != "pass" for row in before), before)

        repaired = annotator.apply_local_caption_density_repair(
            payload,
            rows,
            "cvpair_global_0001",
            caption_variants=2,
        )
        after = annotator.preview_caption_length_qa(repaired, rows, "cvpair_global_0001", caption_variants=2)
        self.assertTrue(all(row["status"] == "pass" for row in after), after)
        self.assertIn("local_caption_density_repair", repaired["qa_notes"])


class AnnotateCvpairTextV4Test(unittest.TestCase):
    def test_program_consensus_requires_both_views_and_rejects_conflict_upgrade(self) -> None:
        rows = make_source_rows("train")
        matching = make_v4_perception(rows, ground_color="gray", uav_color="gray")
        consensus, _ = annotator.finalize_cross_view_consensus(matching, {})
        self.assertEqual(
            [(item["attribute"], item["value"]) for item in consensus["stable"]],
            [("color", "gray")],
        )

        one_view = make_v4_perception(rows, ground_color="gray", uav_color="uncertain")
        consensus, _ = annotator.finalize_cross_view_consensus(one_view, {})
        self.assertFalse(any(item["attribute"] == "color" for item in consensus["stable"]))

        conflicting = make_v4_perception(rows, ground_color="gray", uav_color="black")
        consensus, _ = annotator.finalize_cross_view_consensus(
            conflicting,
            {"stable_suggestions": [{"attribute": "color", "value": "gray"}]},
        )
        self.assertFalse(any(item["attribute"] == "color" for item in consensus["stable"]))
        self.assertTrue(any(item["attribute"] == "color" for item in consensus["conflict"]))
        self.assertEqual(len(consensus["vlm_adjudication"]["ignored_upgrade_suggestions"]), 1)

    def test_vlm_synonym_normalization_cannot_remap_known_canonical_value(self) -> None:
        rows = make_source_rows("train")
        perception = make_v4_perception(rows, ground_color="charcoal", uav_color="gray")
        consensus, _ = annotator.finalize_cross_view_consensus(
            perception,
            {
                "normalization_suggestions": [
                    {
                        "attribute": "color",
                        "raw_value": "charcoal",
                        "canonical_value": "gray",
                        "reason": "charcoal is a gray shade",
                    }
                ]
            },
        )
        self.assertTrue(any(item["attribute"] == "color" for item in consensus["stable"]))

        conflicting = make_v4_perception(rows, ground_color="gray", uav_color="black")
        consensus, _ = annotator.finalize_cross_view_consensus(
            conflicting,
            {
                "normalization_suggestions": [
                    {
                        "attribute": "color",
                        "raw_value": "gray",
                        "canonical_value": "black",
                        "reason": "invalid attempted remap",
                    }
                ]
            },
        )
        self.assertFalse(any(item["attribute"] == "color" for item in consensus["stable"]))
        self.assertTrue(consensus["vlm_adjudication"]["rejected_normalizations"])

    def test_v4_local_qa_enforces_fact_and_uncertainty_coverage(self) -> None:
        rows = make_source_rows("train")
        payload = make_v4_caption_payload(rows)
        local_qa = annotator.run_v4_local_qa(
            payload,
            rows,
            "cvpair_global_0001",
            caption_variants=2,
            evidence_confidence_threshold=0.75,
            id_coverage_threshold=0.8,
            image_coverage_threshold=0.85,
            uncertainty_coverage_threshold=1.0,
        )
        self.assertEqual(local_qa["status"], "pass", local_qa)

        payload["annotation_id"]["caption_claims"]["canonical"].append(
            {
                "claim": "the vehicle has a roof rack",
                "claim_type": "positive",
                "attribute": "roof_feature",
                "value": "roof_rack",
                "fact_ids": ["unknown:roof_rack"],
                "reason": "",
            }
        )
        coverage_rows = annotator.build_fact_coverage_qa_rows(
            payload,
            rows,
            "cvpair_global_0001",
            caption_variants=2,
        )
        canonical_id = next(
            row
            for row in coverage_rows
            if row["level"] == "id" and row["variant"] == "canonical"
        )
        self.assertEqual(canonical_id["status"], "manual_review")
        self.assertIn("unsupported_claim", canonical_id["issues"])

        payload = make_v4_caption_payload(rows)
        payload["cross_view_consensus"]["uncertain"].append(
            {
                "attribute": "roof_feature",
                "values": [],
                "source_images": [row["image_path"] for row in rows],
                "reason": "roof details are too small",
            }
        )
        coverage_rows = annotator.build_fact_coverage_qa_rows(
            payload,
            rows,
            "cvpair_global_0001",
            caption_variants=2,
        )
        canonical_id = next(
            row
            for row in coverage_rows
            if row["level"] == "id" and row["variant"] == "canonical"
        )
        self.assertIn("uncertainty_coverage_below_threshold", canonical_id["issues"])

    def test_claim_level_audit_failure_is_not_hidden_by_overall_score(self) -> None:
        rows = make_source_rows("train")
        id_items = []
        image_items = []
        for variant in annotator.caption_variant_names(2):
            id_items.append(
                {
                    "variant": variant,
                    "all_caption_claims_covered": True,
                    "claims": [
                        {
                            "claim": "the body is gray",
                            "attribute": "color",
                            "value": "gray",
                            "status": "supported",
                            "source_images": [rows[0]["image_path"], rows[2]["image_path"]],
                            "correction": "",
                        }
                    ],
                }
            )
            for row in rows:
                image_items.append(
                    {
                        "image_path": row["image_path"],
                        "variant": variant,
                        "all_caption_claims_covered": True,
                        "claims": [
                            {
                                "claim": "the body is gray",
                                "attribute": "color",
                                "value": "gray",
                                "status": "supported",
                                "source_images": [row["image_path"]],
                                "correction": "",
                            }
                        ],
                    }
                )
        audit = annotator.normalize_claim_audit(
            {"id": id_items, "images": image_items},
            rows,
            "cvpair_global_0001",
            caption_variants=2,
            audit_round=1,
        )
        self.assertEqual(audit["status"], "pass", audit)

        image_items[0]["claims"][0]["status"] = "not_visible"
        image_items[0]["claims"][0]["correction"] = "remove unsupported color claim"
        audit = annotator.normalize_claim_audit(
            {"id": id_items, "images": image_items},
            rows,
            "cvpair_global_0001",
            caption_variants=2,
            audit_round=1,
        )
        self.assertEqual(audit["status"], "manual_review")
        self.assertTrue(
            any(row["status"] == "not_visible" for row in audit["claim_rows"])
        )

    def test_v4_stage_order_repairs_and_reaudits_after_vlm_b_failure(self) -> None:
        rows = make_source_rows("train")
        perception = make_v4_perception(rows)
        caption_payload = make_v4_caption_payload(rows)
        args = SimpleNamespace(
            dataset_root=Path("."),
            model="gpt-4o-mini",
            model_b="qwen2.5-vl-72b-instruct",
            model_a_family=None,
            model_b_family=None,
            base_url="https://vlm-a.invalid/v1",
            audit_base_url="https://vlm-b.invalid/v1",
            max_image_side=1024,
            jpeg_quality=85,
            caption_variants=2,
            evidence_confidence_threshold=0.75,
            id_coverage_threshold=0.8,
            image_coverage_threshold=0.85,
            uncertainty_coverage_threshold=1.0,
            max_new_observation_rounds=1,
            temperature=0.0,
            max_tokens=6000,
            timeout=30.0,
            retries=0,
        )
        called_stages = []
        audit_calls = 0

        def fake_call(stage, *_args, **_kwargs):
            nonlocal audit_calls
            called_stages.append(stage)
            if stage.startswith("1_vlm_a_per_image_perception"):
                return {"perception": perception}, {"stage": stage}
            if stage.startswith("2_vlm_a_consensus_adjudication"):
                return {
                    "normalization_suggestions": [],
                    "conflict_explanations": [],
                    "downgrade_suggestions": [],
                    "stable_suggestions": [],
                }, {"stage": stage}
            if stage.startswith("3_vlm_a_fact_locked_caption"):
                return caption_payload, {"stage": stage}
            if stage == "5_vlm_b_claim_audit_round_1":
                audit_calls += 1
                return make_supported_claim_audit(rows, failed_first_claim=True), {"stage": stage}
            if stage == "6_vlm_a_repair_from_audit_issues":
                return caption_payload, {"stage": stage}
            if stage == "8_vlm_b_final_claim_reaudit":
                audit_calls += 1
                return make_supported_claim_audit(rows), {"stage": stage}
            self.fail(f"Unexpected stage {stage}")

        with patch.object(
            annotator,
            "image_to_data_url",
            return_value="data:image/jpeg;base64,AA==",
        ), patch.object(annotator, "call_json_model_stage", side_effect=fake_call):
            payload, raw = annotator.run_job_v4(
                "cvpair_global_0001",
                rows,
                args,
                api_key="vlm-a-key",
                audit_api_key="vlm-b-key",
            )

        self.assertEqual(
            called_stages,
            [
                "1_vlm_a_per_image_perception",
                "2_vlm_a_consensus_adjudication_round_0",
                "3_vlm_a_fact_locked_caption_round_0",
                "5_vlm_b_claim_audit_round_1",
                "6_vlm_a_repair_from_audit_issues",
                "8_vlm_b_final_claim_reaudit",
            ],
        )
        stage_names = [stage["stage"] for stage in raw["stages"]]
        self.assertLess(
            stage_names.index("4_local_schema_length_fact_coverage_qa"),
            stage_names.index("5_vlm_b_claim_audit_round_1"),
        )
        self.assertLess(
            stage_names.index("7_local_qa_after_vlm_a_repair"),
            stage_names.index("8_vlm_b_final_claim_reaudit"),
        )
        self.assertEqual(audit_calls, 2)
        self.assertTrue(raw["repair_triggered"])
        self.assertEqual(payload["local_qa"]["status"], "pass")
        self.assertEqual(payload["claim_audit"]["status"], "pass")
        self.assertEqual(payload["annotation_id"]["qa_status"], "auto_labeled")

    def test_new_observation_reenters_perception_and_consensus(self) -> None:
        rows = make_source_rows("train")
        perception = make_v4_perception(rows)
        first_caption = copy.deepcopy(make_v4_caption_payload(rows))
        first_caption["new_observations"] = [
            {
                "image_path": rows[0]["image_path"],
                "attribute": "window_feature",
                "proposed_value": "dark_window_area",
                "visual_evidence": "dark side-window region may be visible",
            }
        ]
        final_caption = make_v4_caption_payload(rows)
        args = SimpleNamespace(
            dataset_root=Path("."),
            model="gpt-4o-mini",
            model_b="qwen2.5-vl-72b-instruct",
            model_a_family=None,
            model_b_family=None,
            base_url="https://vlm-a.invalid/v1",
            audit_base_url="https://vlm-b.invalid/v1",
            max_image_side=1024,
            jpeg_quality=85,
            caption_variants=2,
            evidence_confidence_threshold=0.75,
            id_coverage_threshold=0.8,
            image_coverage_threshold=0.85,
            uncertainty_coverage_threshold=1.0,
            max_new_observation_rounds=1,
            temperature=0.0,
            max_tokens=6000,
            timeout=30.0,
            retries=0,
        )
        called_stages = []

        def fake_call(stage, *_args, **_kwargs):
            called_stages.append(stage)
            if stage.startswith("1_vlm_a"):
                return {"perception": perception}, {"stage": stage}
            if stage.startswith("2_vlm_a"):
                return {
                    "normalization_suggestions": [],
                    "conflict_explanations": [],
                    "downgrade_suggestions": [],
                    "stable_suggestions": [],
                }, {"stage": stage}
            if stage == "3_vlm_a_fact_locked_caption_round_0":
                return first_caption, {"stage": stage}
            if stage == "3_vlm_a_fact_locked_caption_round_1":
                return final_caption, {"stage": stage}
            if stage == "5_vlm_b_claim_audit_round_1":
                return make_supported_claim_audit(rows), {"stage": stage}
            self.fail(f"Unexpected stage {stage}")

        with patch.object(
            annotator,
            "image_to_data_url",
            return_value="data:image/jpeg;base64,AA==",
        ), patch.object(annotator, "call_json_model_stage", side_effect=fake_call):
            payload, raw = annotator.run_job_v4(
                "cvpair_global_0001",
                rows,
                args,
                api_key="vlm-a-key",
                audit_api_key="vlm-b-key",
            )

        self.assertIn("1_vlm_a_perception_reentry_round_1", called_stages)
        self.assertIn("2_vlm_a_consensus_adjudication_round_1", called_stages)
        self.assertIn("3_vlm_a_fact_locked_caption_round_1", called_stages)
        self.assertEqual(raw["new_observation_rounds"], 1)
        self.assertEqual(payload["new_observations"], [])

    def test_vlm_b_evidence_input_redacts_vlm_a_confidence(self) -> None:
        rows = make_source_rows("train")
        payload = make_v4_caption_payload(rows)
        local_qa = annotator.run_v4_local_qa(
            payload,
            rows,
            "cvpair_global_0001",
            caption_variants=2,
            evidence_confidence_threshold=0.75,
            id_coverage_threshold=0.8,
            image_coverage_threshold=0.85,
            uncertainty_coverage_threshold=1.0,
        )
        with patch.object(
            annotator,
            "image_to_data_url",
            return_value="data:image/jpeg;base64,AA==",
        ):
            content = annotator.build_v4_claim_audit_content(
                "cvpair_global_0001",
                rows,
                Path("."),
                payload,
                local_qa,
                max_image_side=1024,
                jpeg_quality=85,
                caption_variants=2,
                audit_round=1,
            )
        instruction = json.loads(content[0]["text"])

        def contains_confidence_key(value):
            if isinstance(value, dict):
                return "confidence" in value or any(
                    contains_confidence_key(item) for item in value.values()
                )
            if isinstance(value, list):
                return any(contains_confidence_key(item) for item in value)
            return False

        self.assertFalse(
            contains_confidence_key(instruction["evidence_tables_without_confidence"])
        )
        self.assertFalse(contains_confidence_key(instruction["captions"]))

    def test_training_identity_review_sampling_routes_all_identity_images(self) -> None:
        id_row, image_rows, *_ = run_local_qa_pipeline("train")
        source_rows = make_source_rows("train")
        annotator.apply_human_review_policy(
            id_row,
            image_rows,
            source_rows,
            enforce_eval_review=True,
            train_review_rate=1.0,
            review_sampling_seed=20260629,
        )
        self.assertTrue(id_row["review_required"])
        self.assertIn("sampled_train_identity", id_row["review_reasons"])
        for image_row in image_rows:
            self.assertTrue(image_row["review_required"])
            self.assertIn("sampled_train_identity", image_row["review_reasons"])

    def test_v4_rejects_same_model_family_for_independent_audit(self) -> None:
        args = SimpleNamespace(
            model="gpt-4o-mini",
            model_b="o4-mini",
            model_a_family=None,
            model_b_family=None,
            audit_base_url=None,
            base_url="https://api.invalid/v1",
        )
        with self.assertRaises(SystemExit):
            annotator.validate_v4_api_configuration(args)

    def test_v4_main_writes_all_quality_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir) / "dataset"
            metadata_dir = dataset_root / "metadata"
            output_dir = dataset_root / "annotations" / "v4_smoke"
            metadata_dir.mkdir(parents=True)
            rows = make_source_rows("train")
            annotator.write_jsonl(metadata_dir / "metadata.jsonl", rows)
            perception = make_v4_perception(rows)
            caption_payload = make_v4_caption_payload(rows)
            args = SimpleNamespace(
                dataset_root=dataset_root,
                metadata=None,
                output_dir=output_dir,
                vehicle_id=[],
                limit_ids=1,
                images_per_id=4,
                require_both_views=True,
                model="gpt-4o-mini",
                model_b="qwen2.5-vl-72b-instruct",
                model_a_family=None,
                model_b_family=None,
                base_url="https://vlm-a.invalid/v1",
                api_key_env="V4_TEST_A_KEY",
                audit_base_url="https://vlm-b.invalid/v1",
                audit_api_key_env="V4_TEST_B_KEY",
                temperature=0.0,
                max_tokens=6000,
                max_image_side=1024,
                jpeg_quality=85,
                timeout=30.0,
                retries=0,
                backend="api",
                pipeline="staged_claim_audit",
                caption_variants=2,
                semantic_match_threshold=0.75,
                evidence_confidence_threshold=0.75,
                id_coverage_threshold=0.8,
                image_coverage_threshold=0.85,
                uncertainty_coverage_threshold=1.0,
                max_new_observation_rounds=1,
                human_review_eval=True,
                train_review_rate=0.0,
                review_sampling_seed=20260629,
                repair_caption_length=True,
                mock=False,
                overwrite=True,
            )

            def fake_call(stage, *_args, **_kwargs):
                if stage.startswith("1_vlm_a_per_image_perception"):
                    return {"perception": perception}, {"stage": stage}
                if stage.startswith("2_vlm_a_consensus_adjudication"):
                    return {
                        "normalization_suggestions": [],
                        "conflict_explanations": [],
                        "downgrade_suggestions": [],
                        "stable_suggestions": [],
                    }, {"stage": stage}
                if stage.startswith("3_vlm_a_fact_locked_caption"):
                    return caption_payload, {"stage": stage}
                if stage == "5_vlm_b_claim_audit_round_1":
                    return make_supported_claim_audit(rows), {"stage": stage}
                self.fail(f"Unexpected stage {stage}")

            with patch.object(annotator, "parse_args", return_value=args), patch.object(
                annotator,
                "image_to_data_url",
                return_value="data:image/jpeg;base64,AA==",
            ), patch.object(
                annotator,
                "call_json_model_stage",
                side_effect=fake_call,
            ), patch.dict(
                os.environ,
                {"V4_TEST_A_KEY": "a-key", "V4_TEST_B_KEY": "b-key"},
                clear=False,
            ), redirect_stdout(io.StringIO()):
                annotator.main()

            expected_files = {
                "annotations_id.jsonl",
                "annotations_image.jsonl",
                "annotation_evidence.jsonl",
                "semantic_qa.jsonl",
                "schema_qa.jsonl",
                "caption_length_qa.jsonl",
                "fact_coverage_qa.jsonl",
                "claim_audit.jsonl",
                "claim_audit_summary.jsonl",
                "raw_responses.jsonl",
                "errors.jsonl",
                "run_manifest.json",
                "annotation_smoke_test_report.md",
            }
            self.assertTrue(expected_files.issubset({path.name for path in output_dir.iterdir()}))
            manifest = json.loads((output_dir / "run_manifest.json").read_text())
            self.assertEqual(manifest["error_count"], 0)
            self.assertEqual(manifest["schema_qa_fail_count"], 0)
            self.assertEqual(manifest["fact_coverage_qa_fail_count"], 0)
            self.assertEqual(manifest["claim_audit_fail_count"], 0)
            self.assertEqual(manifest["caption_length_fail_count"], 0)
            id_row = json.loads((output_dir / "annotations_id.jsonl").read_text().splitlines()[0])
            self.assertEqual(id_row["qa_status"], "auto_labeled")


if __name__ == "__main__":
    unittest.main()
