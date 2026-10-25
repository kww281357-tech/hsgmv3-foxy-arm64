import numpy as np
from torch.backends.mkl import verbose
from ultralytics import YOLOE
import cv2
from segmentation.object_list import *
import os
import matplotlib.pyplot as plt
from semantic_evidence import canonical_label, rgb_to_bgr
from task_visual import bounded_instruction_terms, is_equivalent_label


# load model once globally
_DEFAULT_CKPT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ckpt/yoloe-26x-seg.pt"))
MODEL_PATH = _DEFAULT_CKPT if os.path.exists(_DEFAULT_CKPT) else "../ckpt/yoloe-26x-seg.pt"
model = YOLOE(MODEL_PATH)
category_names = {cat['id']: cat['name'] for cat in categories}
_BASE_CLASS_NAMES = tuple(category_names.values())
_class_name_list = list(_BASE_CLASS_NAMES)
_prompt_name_list = [name.replace('_', ' ') for name in _class_name_list]
_BASE_TEXT_PE = model.get_text_pe(_prompt_name_list)
model.set_classes(_class_name_list, _BASE_TEXT_PE)
_ACTIVE_CLASS_NAMES = _BASE_CLASS_NAMES
_ACTIVE_TEXT_PE = _BASE_TEXT_PE
_ACTIVE_TASK_CLASS_IDS = ()
_ACTIVE_TASK_NAMES = ()
_ACTIVE_TASK_TEXT_PE = None
_TEXT_PE_CACHE = {_BASE_CLASS_NAMES: _BASE_TEXT_PE}
_MAX_TEXT_PE_CACHE = 4
_CACHE_MODEL_ID = id(model)
_GENERIC_WALL_DISTRACTORS = frozenset({"clock", "calendar", "painting", "photo_frame", "whiteboard", "poster"})
_NON_BOOST_STRUCTURAL_HEADS = frozenset({
    "door", "doors", "doorway", "archway", "stairs", "stair", "staircase",
    "hallway", "hall", "room", "area", "opening", "threshold", "entrance",
    "exit", "wall", "floor", "ceiling", "house", "people", "person", "two",
})


def _is_boostable_landmark_term(term: str) -> bool:
    clean = str(term or "").replace("_", " ").strip().lower()
    if not clean or clean in _NON_BOOST_STRUCTURAL_HEADS:
        return False
    tokens = [w for w in clean.split() if w]
    if not tokens or tokens[-1] in _NON_BOOST_STRUCTURAL_HEADS:
        return False
    return True


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    inter = int(np.count_nonzero(a & b))
    if inter <= 0:
        return 0.0
    union = int(np.count_nonzero(a | b))
    return float(inter / union) if union > 0 else 0.0


def instance_segmentation(image: np.ndarray):
    """
    Input: image (np.ndarray) - raw image
    Output: List[dict] - per-instance class_id, class_name, mask (same size as input), conf
    """
    bgr = rgb_to_bgr(image)
    results = model.predict(bgr, conf=0.48, iou=0.3, agnostic_nms=True,
                            retina_masks=True, verbose=False)
    result = results[0]
    output = []
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes
        class_names = result.names
        for i in range(masks.shape[0]):
            class_id = int(boxes.cls[i].item())
            conf_val = float(boxes.conf[i].item()) if boxes.conf is not None else 1.0
            class_name = class_names.get(class_id, str(class_id))
            mask = masks[i].astype(np.uint8) * 255
            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            output.append({
                "class_id": class_id,
                "class_name": class_name,
                "canonical_label": canonical_label(class_name),
                "mask": mask_resized,
                "conf": conf_val
            })

    # Task-conditioned open-vocabulary boost pass: prevent 240 generic base classes
    # (such as clock/calendar/painting under agnostic_nms=True and cls.max(1) argmax)
    # and the 0.48 threshold from suppressing instruction-specific landmarks (e.g. eye chart, white statue).
    task_ids = tuple(globals().get("_ACTIVE_TASK_CLASS_IDS") or ())
    task_names = tuple(globals().get("_ACTIVE_TASK_NAMES") or ())
    task_pe = globals().get("_ACTIVE_TASK_TEXT_PE")
    full_names = tuple(globals().get("_ACTIVE_CLASS_NAMES") or globals().get("_BASE_CLASS_NAMES") or ())
    full_pe = globals().get("_ACTIVE_TEXT_PE")
    _os = globals().get("os")
    boost_enabled = (
        str(_os.environ.get("HSGM_YOLOE_OPENVOCAB_BOOST", "1")).strip().lower() not in ("0", "false", "no", "off")
        if _os is not None
        else True
    )
    if boost_enabled and task_ids and hasattr(model, "model"):
        try:
            use_task_slice = bool(
                task_names
                and len(task_names) == len(task_ids)
                and task_pe is not None
                and full_pe is not None
                and hasattr(model, "set_classes")
            )
            try:
                if use_task_slice:
                    model.set_classes(list(task_names), task_pe)
                    if getattr(model, "predictor", None) is not None and hasattr(model.predictor, "model"):
                        model.predictor.model.names = model.model.names
                    supp_results = model.predict(
                        bgr,
                        conf=0.22,
                        iou=0.45,
                        agnostic_nms=False,
                        retina_masks=True,
                        verbose=False,
                    )
                else:
                    supp_results = model.predict(
                        bgr,
                        conf=0.22,
                        iou=0.45,
                        agnostic_nms=False,
                        classes=list(task_ids),
                        retina_masks=True,
                        verbose=False,
                    )
            finally:
                if use_task_slice:
                    model.set_classes(list(full_names), full_pe)
                    if getattr(model, "predictor", None) is not None and hasattr(model.predictor, "model"):
                        model.predictor.model.names = model.model.names
            supp_res = supp_results[0] if supp_results else None
            if supp_res is not None and supp_res.masks is not None:
                s_masks = supp_res.masks.data.cpu().numpy()
                s_boxes = supp_res.boxes
                s_names = supp_res.names
                task_id_set = set(task_ids)
                for i in range(s_masks.shape[0]):
                    raw_cid = int(s_boxes.cls[i].item())
                    if use_task_slice and 0 <= raw_cid < len(task_ids):
                        cid = int(task_ids[raw_cid])
                        cname = str(task_names[raw_cid])
                    else:
                        cid = raw_cid
                        cname = s_names.get(cid, str(cid))
                    raw_conf = float(s_boxes.conf[i].item()) if s_boxes.conf is not None else 0.5
                    canon = canonical_label(cname)
                    smask = s_masks[i].astype(np.uint8) * 255
                    smask_resized = cv2.resize(smask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                    if not np.any(smask_resized > 0):
                        continue
                    already_present = any(
                        existing.get("canonical_label") == canon and _mask_iou(existing["mask"], smask_resized) >= 0.35
                        for existing in output
                    )
                    if already_present:
                        continue
                    # Remove generic non-task wall distractors (e.g. clock/calendar) that suppressed this task landmark
                    output = [
                        existing for existing in output
                        if not (
                            existing.get("class_id") not in task_id_set
                            and existing.get("canonical_label") in _GENERIC_WALL_DISTRACTORS
                            and _mask_iou(existing["mask"], smask_resized) >= 0.45
                        )
                    ]
                    calibrated_conf = float(max(raw_conf, min(0.92, 0.50 + 0.55 * raw_conf)))
                    output.append({
                        "class_id": cid,
                        "class_name": cname,
                        "canonical_label": canon,
                        "mask": smask_resized,
                        "conf": calibrated_conf,
                        "raw_conf": raw_conf,
                    })
        except Exception:
            pass
    return output

def get_class_color(class_id: int) -> tuple:
    # use matplotlib tab20 colormap for fixed, distinct per-class colors
    color_map = plt.get_cmap('tab20')
    color = color_map(class_id % 20)[:3]  # take first 3 channels (RGB)
    return tuple(int(255 * c) for c in color)

def visualize_instance_segmentation(image: np.ndarray, instances: list, alpha: float = 0.5) -> np.ndarray:
    vis_image = image.copy()
    if vis_image.shape[2] == 4:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGBA2BGR)
    for inst in instances:
        class_id = inst["class_id"]
        color = get_class_color(class_id)
        mask = inst["mask"]
        colored_mask = np.zeros_like(vis_image)
        for c in range(3):
            colored_mask[:, :, c] = mask // 255 * color[c]
        vis_image = cv2.addWeighted(vis_image, 1, colored_mask, alpha, 0)
        ys, xs = np.where(mask > 0)
        if len(xs) > 0 and len(ys) > 0:
            center_x, center_y = int(xs.mean()), int(ys.mean())
            center_x = np.clip(center_x, 0, vis_image.shape[1] - 1)
            center_y = np.clip(center_y, 0, vis_image.shape[0] - 1)
            cv2.putText(vis_image, inst["class_name"], (center_x, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 4, cv2.LINE_AA)
            cv2.putText(vis_image, inst["class_name"], (center_x, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    return vis_image


def configure_task_vocabulary(instruction, enabled: bool = True, max_extra_terms: int = 16) -> dict:
    g = globals()
    base_names = g.get("_BASE_CLASS_NAMES")
    if not base_names:
        base_names = tuple(g.get("_class_name_list") or [cat["name"] for cat in g.get("categories", ())])
        g["_BASE_CLASS_NAMES"] = base_names
    cache = g.setdefault("_TEXT_PE_CACHE", {})
    if g.get("_CACHE_MODEL_ID") != id(model):
        cache.clear()
        g["_ACTIVE_CLASS_NAMES"] = base_names
        g["_CACHE_MODEL_ID"] = id(model)
    active_names = g.get("_ACTIVE_CLASS_NAMES", base_names)
    limit = max(0, int(max_extra_terms))
    extra_terms = []
    bounded_terms = bounded_instruction_terms(instruction, max_extra_terms=max(limit * 2, 32)) if (enabled and limit > 0) else []
    if enabled and limit > 0:
        for term in bounded_terms:
            if not is_equivalent_label(term, base_names) and not is_equivalent_label(term, extra_terms):
                extra_terms.append(term)
                if len(extra_terms) >= limit:
                    break
    target_names = base_names + tuple(extra_terms)
    task_class_ids = []
    boost_terms = [t for t in bounded_terms if _is_boostable_landmark_term(t)] if (enabled and bounded_terms) else []
    if boost_terms:
        for idx, bname in enumerate(base_names):
            if _is_boostable_landmark_term(bname) and any(is_equivalent_label(term, (bname,)) for term in boost_terms):
                task_class_ids.append(idx)
        for idx in range(len(base_names), len(target_names)):
            if _is_boostable_landmark_term(target_names[idx]):
                task_class_ids.append(idx)
    g["_ACTIVE_TASK_CLASS_IDS"] = tuple(task_class_ids)
    g["_ACTIVE_TASK_NAMES"] = tuple(target_names[i] for i in task_class_ids)
    changed = target_names != active_names
    if changed:
        prompts = [name.replace("_", " ") for name in target_names]
        text_pe = cache.pop(target_names) if target_names in cache else model.get_text_pe(prompts)
        cache[target_names] = text_pe
        max_cache = int(g.get("_MAX_TEXT_PE_CACHE", 4))
        while len(cache) > max_cache:
            cache.pop(next(iter(cache)))
        model.set_classes(list(target_names), text_pe)
        if getattr(model, "predictor", None) is not None and hasattr(model.predictor, "model"):
            model.predictor.model.names = model.model.names
        g["_class_name_list"] = list(target_names)
        g["_prompt_name_list"] = prompts
        g["_ACTIVE_CLASS_NAMES"] = target_names
        g["_ACTIVE_TEXT_PE"] = text_pe
    elif target_names in cache:
        text_pe = cache.pop(target_names)
        cache[target_names] = text_pe
        g["_ACTIVE_TEXT_PE"] = text_pe
    else:
        text_pe = g.get("_ACTIVE_TEXT_PE")
    try:
        g["_ACTIVE_TASK_TEXT_PE"] = (
            text_pe[:, task_class_ids, :]
            if (task_class_ids and text_pe is not None and hasattr(text_pe, "__getitem__"))
            else None
        )
    except Exception:
        g["_ACTIVE_TASK_TEXT_PE"] = None
    return {
        "enabled": bool(enabled),
        "active_labels": list(target_names),
        "extra_terms": extra_terms,
        "changed": changed,
        "cache_size": len(cache),
    }

