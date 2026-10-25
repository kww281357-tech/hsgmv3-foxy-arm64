"""Verifier-stage open-vocabulary detection and segmentation using GroundingDINO + SAM3."""
from __future__ import annotations

from contextlib import contextmanager
import logging
import os
import sys
import time
import types
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision.ops import nms

from segmentation.object_list import categories
from semantic_evidence import canonical_label, rgb_to_bgr
from task_visual import bounded_instruction_terms, is_equivalent_label

_BASE_CATEGORY_ID_BY_CANONICAL: dict[str, int] = {}
for _cat in categories:
    _cname = canonical_label(_cat.get("name", ""))
    if _cname and _cname not in _BASE_CATEGORY_ID_BY_CANONICAL:
        _BASE_CATEGORY_ID_BY_CANONICAL[_cname] = int(_cat.get("id", 0))

_CORE_VERIFIER_TERMS: tuple[str, ...] = (
    "door",
    "doorway",
    "stairs",
    "railing",
    "statue",
    "eye chart",
    "picture",
    "mirror",
    "window",
    "chair",
    "table",
    "couch",
    "bed",
    "cabinet",
    "counter",
    "sink",
    "toilet",
    "bathtub",
    "shower",
    "refrigerator",
    "tv_monitor",
    "plant",
    "curtain",
    "shelving",
    "fireplace",
    "chest_of_drawers",
    "stool",
    "towel",
)

_DEFAULT_CKPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ckpt"))
_DINO_MODEL_CACHE: dict[str, Any] = {}
_SAM3_PREDICTOR_CACHE: dict[str, Any] = {}


def verifier_dino_sam3_enabled(cfg: Any = None, env: Any = None) -> bool:
    """Return True when verifier-stage DINO + SAM3 perception is enabled via config or environment."""
    env_map = os.environ if env is None else env
    raw_env = str(env_map.get("HSGM_VERIFIER_DINO_SAM3", "") or "").strip().lower()
    if raw_env in ("1", "true", "yes", "on"):
        return True
    if raw_env in ("0", "false", "no", "off"):
        return False
    stack_env = str(env_map.get("HSGM_VERIFIER_PERCEPTION_STACK", "") or "").strip().lower()
    if stack_env in ("dino_sam3", "dinosam3", "grounding_dino_sam3"):
        return True
    if stack_env in ("current", "yoloe", "none", "off"):
        return False
    if isinstance(cfg, dict):
        return bool(cfg.get("verifier_dino_sam3_enabled", False))
    return False


def _ensure_sam3_runtime_shims() -> None:
    """Provide PyTorch 2.0 compatibility shims required by Ultralytics SAM3 modules without pip auto-install."""
    if "torch.nn.attention" not in sys.modules:
        attn_mod = types.ModuleType("torch.nn.attention")

        class SDPBackend:
            FLASH_ATTENTION = "flash"
            EFFICIENT_ATTENTION = "mem_efficient"
            MATH = "math"
            CUDNN_ATTENTION = "cudnn"

        @contextmanager
        def sdpa_kernel(backends):
            yield

        attn_mod.SDPBackend = SDPBackend
        attn_mod.sdpa_kernel = sdpa_kernel
        sys.modules["torch.nn.attention"] = attn_mod
        setattr(torch.nn, "attention", attn_mod)

    if "timm" not in sys.modules:
        timm_mod = types.ModuleType("timm")
        layers_mod = types.ModuleType("timm.layers")

        class DropPath(nn.Module):
            def __init__(self, drop_prob: float = 0.0):
                super().__init__()
                self.drop_prob = float(drop_prob)

            def forward(self, x):
                return x

        class Mlp(nn.Module):
            def __init__(
                self,
                in_features: int,
                hidden_features: int | None = None,
                out_features: int | None = None,
                act_layer=nn.GELU,
                norm_layer=None,
                bias: bool = True,
                drop: float | tuple[float, float] = 0.0,
            ):
                super().__init__()
                out_f = out_features or in_features
                hid_f = hidden_features or in_features
                drop_probs = drop if isinstance(drop, tuple) else (float(drop), float(drop))
                self.fc1 = nn.Linear(in_features, hid_f, bias=bias)
                self.act = act_layer()
                self.drop1 = nn.Dropout(drop_probs[0])
                self.norm = norm_layer(hid_f) if norm_layer is not None else nn.Identity()
                self.fc2 = nn.Linear(hid_f, out_f, bias=bias)
                self.drop2 = nn.Dropout(drop_probs[1])

            def forward(self, x):
                x = self.fc1(x)
                x = self.act(x)
                x = self.drop1(x)
                x = self.norm(x)
                x = self.fc2(x)
                x = self.drop2(x)
                return x

        layers_mod.DropPath = DropPath
        layers_mod.Mlp = Mlp
        timm_mod.layers = layers_mod
        sys.modules["timm"] = timm_mod
        sys.modules["timm.layers"] = layers_mod

    import ultralytics.models.sam.sam3.vitdet as vd

    vd.check_requirements = lambda *a, **kw: True


def _load_sam3_interactive_checkpoint(model: nn.Module, checkpoint_path: str) -> nn.Module:
    """Load interactive SAM3 / SAM3.1 checkpoint tensors (supporting both FP16 and FP32 state dicts)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and isinstance(ckpt.get("model"), dict):
        ckpt = ckpt["model"]
    norm_ckpt = {
        k.replace("tracker.model.interactive_", "tracker.")
        .replace("tracker.model.interactivity_", "tracker.")
        .replace("interactive_convs", "sam2_convs"): (
            v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
        )
        for k, v in ckpt.items()
        if isinstance(v, torch.Tensor)
    }
    sam3_image_ckpt = {k.replace("detector.", ""): v for k, v in norm_ckpt.items() if "detector" in k}
    sam3_image_ckpt.update(
        {
            k.replace("backbone.vision_backbone", "image_encoder.vision_backbone"): v
            for k, v in sam3_image_ckpt.items()
            if "backbone.vision_backbone" in k
        }
    )
    sam3_image_ckpt.update(
        {
            k.replace("tracker.transformer.encoder", "memory_attention"): v
            for k, v in norm_ckpt.items()
            if "tracker.transformer" in k
        }
    )
    sam3_image_ckpt.update(
        {
            k.replace("tracker.maskmem_backbone", "memory_encoder"): v
            for k, v in norm_ckpt.items()
            if "tracker.maskmem_backbone" in k
        }
    )
    sam3_image_ckpt.update({k.replace("tracker.", ""): v for k, v in norm_ckpt.items() if k.startswith("tracker.")})
    model_sd = model.state_dict()
    filtered = {k: v for k, v in sam3_image_ckpt.items() if k in model_sd and model_sd[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model


def _get_dino_model(dino_ckpt_dir: str | None = None):
    ckpt_dir = dino_ckpt_dir or os.environ.get(
        "HSGM_DINO_CKPT_DIR", os.path.join(_DEFAULT_CKPT_DIR, "grounding-dino-base")
    )
    if ckpt_dir not in _DINO_MODEL_CACHE:
        from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = GroundingDinoProcessor.from_pretrained(ckpt_dir, local_files_only=True)
        model = GroundingDinoForObjectDetection.from_pretrained(ckpt_dir, local_files_only=True).to(device).eval()
        _DINO_MODEL_CACHE[ckpt_dir] = (processor, model, device)
    return _DINO_MODEL_CACHE[ckpt_dir]


def _get_sam3_predictor(sam3_ckpt_path: str | None = None):
    ckpt_path = sam3_ckpt_path or os.environ.get(
        "HSGM_SAM3_CKPT_PATH", os.path.join(_DEFAULT_CKPT_DIR, "sam3.pt")
    )
    if ckpt_path not in _SAM3_PREDICTOR_CACHE:
        _ensure_sam3_runtime_shims()
        import ultralytics.models.sam.build_sam3 as b3
        from ultralytics.models.sam.predict import SAM3Predictor

        orig_loader = b3._load_checkpoint
        try:
            b3._load_checkpoint = lambda model, ckpt, interactive=False: _load_sam3_interactive_checkpoint(
                model, ckpt_path
            )
            sam3_model = b3.build_interactive_sam3(ckpt_path, compile=False, with_backbone=True)
        finally:
            b3._load_checkpoint = orig_loader
        device = "cuda" if torch.cuda.is_available() else "cpu"
        predictor = SAM3Predictor(
            overrides=dict(
                model=ckpt_path,
                task="segment",
                mode="predict",
                imgsz=1008,
                conf=0.0,
                save=False,
                verbose=False,
                device=device,
            )
        )
        predictor.setup_model(model=sam3_model, verbose=False)
        _SAM3_PREDICTOR_CACHE[ckpt_path] = predictor
    return _SAM3_PREDICTOR_CACHE[ckpt_path]


def build_verifier_dino_prompt_terms(
    instruction_text: Any = "",
    endpoint_text: Any = "",
    turn_text: Any = "",
    extra_terms: list[str] | tuple[str, ...] | None = None,
    max_terms: int = 16,
    *,
    task_only: bool = False,
) -> list[str]:
    """Assemble task-prioritized open-vocabulary prompt terms for GroundingDINO at verification time."""
    limit = max(2, min(32, int(max_terms)))
    terms: list[str] = []

    def _add_term(raw: Any) -> None:
        if len(terms) >= limit:
            return
        text = str(raw or "").replace("_", " ").strip().lower()
        if not text:
            return
        if not is_equivalent_label(text, terms):
            terms.append(text)

    for src, cap in ((endpoint_text, 10), (turn_text, 8), (instruction_text, 16)):
        if isinstance(src, str) and src.strip():
            for t in bounded_instruction_terms(src, max_extra_terms=cap):
                _add_term(t)
    for t in extra_terms or ():
        _add_term(t)
    combined_lower = f"{endpoint_text} {turn_text} {instruction_text}".lower()
    if any(w in combined_lower for w in ("door", "doorway", "entrance", "archway")):
        _add_term("doorway")
        _add_term("door")
    if any(w in combined_lower for w in ("statue", "sculpture")):
        _add_term("statue")
    if any(w in combined_lower for w in ("stair", "stairs", "staircase", "step", "steps")):
        _add_term("stairs")
    if not task_only or not terms:
        for t in _CORE_VERIFIER_TERMS:
            _add_term(t)
    return terms[:limit]


def _match_prompt_term(raw_label: str, prompt_terms: list[str]) -> str:
    clean = str(raw_label or "").replace("_", " ").strip().lower()
    if not clean:
        return prompt_terms[0] if prompt_terms else "object"
    for term in prompt_terms:
        if is_equivalent_label(clean, (term,)):
            return term
    for term in prompt_terms:
        t_clean = term.replace("_", " ").strip().lower()
        if t_clean in clean or clean in t_clean:
            return term
    return clean


def _class_id_for_term(term: str, prompt_terms: list[str]) -> int:
    canon = canonical_label(term)
    if canon in _BASE_CATEGORY_ID_BY_CANONICAL:
        return _BASE_CATEGORY_ID_BY_CANONICAL[canon]
    base_offset = max(_BASE_CATEGORY_ID_BY_CANONICAL.values(), default=100) + 1
    for idx, pt in enumerate(prompt_terms):
        if canonical_label(pt) == canon:
            return base_offset + idx
    return base_offset + len(prompt_terms)


def dino_sam3_segment_view(
    image: np.ndarray,
    prompt_terms: list[str] | tuple[str, ...] | None = None,
    *,
    box_threshold: float = 0.20,
    text_threshold: float = 0.16,
    max_detections: int = 12,
    dino_ckpt_dir: str | None = None,
    sam3_ckpt_path: str | None = None,
) -> list[dict[str, Any]]:
    """Run GroundingDINO + SAM3 on a single RGB view and return instance dicts matching instance_segmentation()."""
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[0] <= 0 or arr.shape[1] <= 0 or arr.shape[2] not in (3, 4):
        return []
    rgb = np.ascontiguousarray(arr[:, :, :3], dtype=np.uint8)
    h, w = rgb.shape[:2]
    terms = list(prompt_terms) if prompt_terms else list(_CORE_VERIFIER_TERMS[:24])
    terms = [str(t).replace("_", " ").strip() for t in terms if str(t).strip()]
    if not terms:
        return []

    processor, dino_model, device = _get_dino_model(dino_ckpt_dir)
    prompt_str = " . ".join(terms) + " ."
    pil_img = Image.fromarray(rgb, mode="RGB")
    inputs = processor(images=pil_img, text=prompt_str, return_tensors="pt")
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = dino_model(**inputs)
    det = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        box_threshold=float(box_threshold),
        text_threshold=float(text_threshold),
        target_sizes=[(h, w)],
    )[0]

    boxes = det.get("boxes")
    scores = det.get("scores")
    raw_labels = det.get("text_labels") if "text_labels" in det else det.get("labels", [])
    if boxes is None or scores is None or len(boxes) == 0:
        return []

    boxes_t = boxes.detach().cpu().float()
    scores_t = scores.detach().cpu().float()
    boxes_t[:, 0].clamp_(0.0, float(w - 1))
    boxes_t[:, 1].clamp_(0.0, float(h - 1))
    boxes_t[:, 2].clamp_(1.0, float(w))
    boxes_t[:, 3].clamp_(1.0, float(h))
    valid_box = (boxes_t[:, 2] - boxes_t[:, 0] >= 4.0) & (boxes_t[:, 3] - boxes_t[:, 1] >= 4.0) & torch.isfinite(scores_t)
    valid_indices = torch.nonzero(valid_box, as_tuple=False).flatten().tolist()
    if not valid_indices:
        return []

    boxes_t = boxes_t[valid_indices]
    scores_t = scores_t[valid_indices]
    labels_list = [str(raw_labels[i]) for i in valid_indices]

    keep_nms = nms(boxes_t, scores_t, 0.50).tolist()
    keep_nms = keep_nms[: max(1, int(max_detections))]
    if not keep_nms:
        return []

    kept_boxes = boxes_t[keep_nms].numpy()
    kept_scores = scores_t[keep_nms].numpy()
    kept_labels = [_match_prompt_term(labels_list[i], terms) for i in keep_nms]

    predictor = _get_sam3_predictor(sam3_ckpt_path)
    bgr = rgb_to_bgr(rgb)
    with torch.no_grad():
        predictor.set_image(bgr)
        sam_results = predictor(bboxes=kept_boxes.tolist())

    if not sam_results or sam_results[0].masks is None:
        return []
    sam_res = sam_results[0]
    masks_np = sam_res.masks.data.detach().cpu().numpy()
    sam_confs = (
        sam_res.boxes.conf.detach().cpu().numpy()
        if (sam_res.boxes is not None and sam_res.boxes.conf is not None)
        else np.ones(len(kept_boxes), dtype=np.float32)
    )

    instances: list[dict[str, Any]] = []
    count = min(len(kept_boxes), masks_np.shape[0])
    for idx in range(count):
        mask_bin = (masks_np[idx] > 0).astype(np.uint8) * 255
        if mask_bin.shape[0] != h or mask_bin.shape[1] != w:
            mask_bin = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)
        if not np.any(mask_bin > 0):
            continue
        term = kept_labels[idx]
        dino_s = float(kept_scores[idx])
        sam_s = float(sam_confs[idx]) if idx < len(sam_confs) and np.isfinite(sam_confs[idx]) else dino_s
        # Combine GroundingDINO detection score and SAM3 mask IoU score so high-quality masks
        # pass mapper close-range confidence gates while preserving ranking order.
        conf_val = float(min(0.99, max(dino_s, 0.5 * dino_s + 0.5 * max(0.0, min(1.0, sam_s)))))
        cls_id = _class_id_for_term(term, terms)
        instances.append(
            {
                "class_id": int(cls_id),
                "class_name": str(term),
                "canonical_label": canonical_label(term),
                "mask": mask_bin,
                "conf": conf_val,
                "dino_conf": dino_s,
                "sam3_conf": sam_s,
                "detector_source": "dino_sam3",
                "dino_box_xyxy": [float(v) for v in kept_boxes[idx].tolist()],
            }
        )
    return instances


_STRUCTURAL_GENERIC_TERMS = frozenset(
    {
        "door",
        "doorway",
        "archway",
        "stairs",
        "stair",
        "staircase",
        "hallway",
        "room",
        "opening",
        "threshold",
        "entrance",
        "exit",
        "people",
        "person",
        "two",
        "wall",
        "floor",
        "ceiling",
        " bathroom",
        "bathroom",
        "bedroom",
        "kitchen",
        "living room",
        "dining room",
        "sitting area",
        "dining area",
        "dinning area",
        "house",
        "hall",
    }
)


def _is_structural_or_generic_term(term: str) -> bool:
    clean = str(term or "").replace("_", " ").strip().lower()
    if not clean or clean in _STRUCTURAL_GENERIC_TERMS:
        return True
    tokens = [w for w in clean.split() if w]
    if not tokens:
        return True
    if tokens[-1] in _STRUCTURAL_GENERIC_TERMS or tokens[-1].rstrip("s") in _STRUCTURAL_GENERIC_TERMS:
        return True
    if all(w in _STRUCTURAL_GENERIC_TERMS for w in tokens):
        return True
    return False


def _missing_task_terms(terms: list[str], current_candidates: Any, obs_id: Any) -> list[str]:
    """Return only those non-structural task prompt terms that are NOT already detected in current_candidates."""
    from target_identity import detector_matches_endpoint

    filtered_terms = [
        t for t in (terms or [])
        if not _is_structural_or_generic_term(t)
    ]
    if not filtered_terms:
        return []
    cand_map = current_candidates if isinstance(current_candidates, dict) else {}
    detected_labels: list[str] = []
    for rec in cand_map.values():
        if not isinstance(rec, dict):
            continue
        if obs_id is not None and rec.get("observation_id") not in (None, obs_id):
            continue
        conf = rec.get("confidence")
        if isinstance(conf, (int, float)) and not isinstance(conf, bool) and float(conf) < 0.45:
            continue
        for k in ("detector_class_name", "canonical_label", "class_name", "label"):
            val = str(rec.get(k) or "").strip()
            if val and val not in detected_labels:
                detected_labels.append(val)
    if not detected_labels:
        return list(filtered_terms)
    missing: list[str] = []
    for term in filtered_terms:
        already_found = is_equivalent_label(term, detected_labels) or any(
            detector_matches_endpoint(lbl, term, "primary") for lbl in detected_labels
        )
        if not already_found:
            missing.append(term)
    return missing


def _active_clause_scoped_texts(
    agent: Any,
    inst_text: str,
    endpoint_text: str,
    turn_text: str,
    ctx_or_text: Any,
) -> tuple[str, str, str]:
    """Scope DINO+SAM3 supplement to active + next route clause so future-room objects are not searched early."""
    inst_obj = getattr(agent, "instruction_obj", None)
    if inst_obj is None or not callable(getattr(inst_obj, "get_current_subtask", None)):
        return inst_text, endpoint_text, turn_text
    try:
        from navigation_contracts import completion_conditions

        key, st = inst_obj.get_current_subtask()
        st = st or {}
        raw_inst = str(st.get("instruction") or inst_text or "")
        checks = completion_conditions(raw_inst, str(key or "P1"))
        if len(checks) <= 2:
            return inst_text, endpoint_text, turn_text
        saved = {
            c["id"]: c
            for c in ((st.get("route_progress") or {}).get("conditions") or [])
            if isinstance(c, dict) and "id" in c
        }
        unsatisfied = [
            c for c in checks
            if saved.get(c["id"], c).get("status") != "satisfied"
        ]
        if not unsatisfied:
            return inst_text, endpoint_text, turn_text
        active_window = unsatisfied[:2]
        active_ids = {c["id"] for c in active_window}
        scoped_inst = " ".join(str(c.get("text") or "") for c in active_window).strip()
        is_completion_proposal = isinstance(ctx_or_text, dict) and ctx_or_text.get("trigger") == "completion_proposal"
        endpoint_active = (checks[-1]["id"] in active_ids) or is_completion_proposal
        scoped_endpoint = endpoint_text if endpoint_active else ""
        return scoped_inst or inst_text, scoped_endpoint, turn_text
    except Exception:
        return inst_text, endpoint_text, turn_text


def refresh_verifier_candidates_with_dino_sam3(
    agent: Any,
    ctx_or_text: Any = None,
    *,
    only_non_structural: bool = False,
) -> dict[str, Any]:
    """Supplement YOLOE candidates with GroundingDINO + SAM3 ONLY for task terms missing in the current observation."""
    cfg = getattr(agent, "navigation_cfg", None) or {}
    if not verifier_dino_sam3_enabled(cfg):
        return {"enabled": False, "refreshed": False}
    mapper = getattr(agent, "mapper", None)
    if mapper is None or not hasattr(mapper, "refresh_current_views_with_segmenter"):
        return {"enabled": True, "refreshed": False, "reason": "mapper_unavailable"}

    obs_id = getattr(mapper, "observation_id", None)
    if obs_id is not None and getattr(mapper, "_last_dino_sam3_observation_id", None) == obs_id:
        cached = getattr(mapper, "_last_dino_sam3_summary", None)
        if isinstance(cached, dict):
            return {**cached, "cached": True}

    endpoint_text = ""
    if hasattr(agent, "_endpoint_clause_text") and callable(getattr(agent, "_endpoint_clause_text")):
        endpoint_text = str(agent._endpoint_clause_text(ctx_or_text) or "")
    elif isinstance(ctx_or_text, str):
        endpoint_text = ctx_or_text
    elif isinstance(ctx_or_text, dict):
        endpoint_text = str(ctx_or_text.get("target_text") or "")

    turn_text = ""
    if hasattr(agent, "_active_turn_clause_info") and callable(getattr(agent, "_active_turn_clause_info")):
        _, turn_text, _ = agent._active_turn_clause_info(ctx_or_text)

    inst_text = str(getattr(agent, "instruction", "") or "")
    scoped_inst, scoped_endpoint, scoped_turn = _active_clause_scoped_texts(
        agent, inst_text, endpoint_text, turn_text, ctx_or_text
    )
    all_terms = build_verifier_dino_prompt_terms(
        instruction_text=scoped_inst,
        endpoint_text=scoped_endpoint,
        turn_text=scoped_turn,
        max_terms=8,
        task_only=True,
    )
    if only_non_structural:
        all_terms = [t for t in all_terms if t.strip().lower() not in _STRUCTURAL_GENERIC_TERMS]
    terms = _missing_task_terms(all_terms, getattr(mapper, "current_visual_candidates", None), obs_id)
    if not terms:
        summary = {
            "enabled": True,
            "refreshed": False,
            "cached": False,
            "observation_id": obs_id,
            "all_task_terms": all_terms,
            "prompt_terms": [],
            "reason": "all_task_terms_detected_by_yoloe",
            "candidate_count": len(getattr(mapper, "current_visual_candidates", None) or {}),
            "elapsed_s": 0.0,
        }
        if not only_non_structural:
            mapper._last_dino_sam3_observation_id = obs_id
            mapper._last_dino_sam3_summary = summary
        return summary

    t0 = time.time()
    view_counts: dict[str, int] = {}

    def _segmenter(rgb: np.ndarray, view_name: str = "unknown") -> list[dict[str, Any]]:
        vname = str(view_name or "").strip().lower()
        if vname == "back":
            dets = dino_sam3_segment_view(
                rgb,
                terms,
                box_threshold=0.45,
                text_threshold=0.28,
                max_detections=2,
            )
        else:
            dets = dino_sam3_segment_view(
                rgb,
                terms,
                box_threshold=0.35,
                text_threshold=0.25,
                max_detections=2,
            )
        view_counts[str(view_name)] = len(dets)
        for d in dets:
            d["detector_source"] = "dino_sam3"
        return dets

    mapper.refresh_current_views_with_segmenter(_segmenter)
    for cand in (getattr(mapper, "current_visual_candidates", None) or {}).values():
        if isinstance(cand, dict) and not cand.get("detector_source"):
            cand["detector_source"] = "dino_sam3"

    elapsed = round(time.time() - t0, 4)
    cand_count = len(getattr(mapper, "current_visual_candidates", None) or {})
    summary = {
        "enabled": True,
        "refreshed": True,
        "cached": False,
        "observation_id": obs_id,
        "all_task_terms": all_terms,
        "prompt_terms": terms,
        "view_detection_counts": view_counts,
        "candidate_count": cand_count,
        "elapsed_s": elapsed,
    }
    mapper._last_dino_sam3_observation_id = obs_id
    mapper._last_dino_sam3_summary = summary
    logging.info(
        f"[VerifierDINO+SAM3] obs_id={obs_id} missing_terms={terms} candidates={cand_count} views={view_counts} elapsed={elapsed:.3f}s"
    )
    return summary

