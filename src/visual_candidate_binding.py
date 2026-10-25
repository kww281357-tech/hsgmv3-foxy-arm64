"""CPU helpers for deterministic visual candidate binding and high-res RGB crops."""
import math
import numpy as np
from PIL import Image

try:
    from src import semantic_evidence, target_identity
except ImportError:
    import semantic_evidence
    import target_identity

VALID_VIEWS = ('front', 'right', 'back', 'left')
_ROLE_PRIORITY = {'primary': 0, 'reference': 1, 'spatial': 2, 'turn': 3}
_MAX_CROP_LIMIT = 6


def _is_strict_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_float(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError, TypeError):
        return False


def _parse_image_size(image_size):
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        return None
    width, height = image_size
    if not (_is_strict_int(width) and _is_strict_int(height)) or width <= 0 or height <= 0:
        return None
    return int(width), int(height)


def _parse_norm_bbox(bbox):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    if not all(_is_finite_float(v) for v in bbox):
        return None
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if x1 < 0.0 or y1 < 0.0 or x2 > 1.0 or y2 > 1.0 or x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _bbox_to_px(bbox, width, height):
    x1, y1, x2, y2 = bbox
    px1 = max(0, min(width, int(math.floor(x1 * width + 1e-6))))
    py1 = max(0, min(height, int(math.floor(y1 * height + 1e-6))))
    px2 = max(0, min(width, int(math.ceil(x2 * width - 1e-6))))
    py2 = max(0, min(height, int(math.ceil(y2 * height - 1e-6))))
    return px1, py1, px2, py2


def _valid_candidate_id(candidate_id, observation_id, view):
    if not isinstance(candidate_id, str) or candidate_id != candidate_id.strip() or not candidate_id:
        return False
    parts = candidate_id.split(':')
    return (len(parts) == 3 and parts[0] == str(observation_id)
            and parts[1] == view and parts[2].isdigit())


def _invalid_resolution(reason, candidate_id=None):
    return {
        'valid': False,
        'reason': str(reason),
        'candidate_id': candidate_id if isinstance(candidate_id, str) else None,
        'entity_id': None,
        'view': None,
        'bbox': None,
        'bbox_space': None,
        'mask': None,
        'mask_key': None,
        'mask_view': None,
        'mask_observation_id': None,
        'image_size': None,
        'confidence': None,
        'selected_detector_class_name': None,
        'canonical_label': None,
    }


def resolve_candidate_region(region, *, candidates, masks, observation_id, endpoint_text, evidence_role='primary'):
    """Bind a VLM-selected candidate_id (and optional full-view sub-bbox) to current detector evidence."""
    if not isinstance(region, dict):
        return _invalid_resolution('invalid_region_payload')
    candidate_id = region.get('candidate_id')
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        return _invalid_resolution('missing_candidate_id', candidate_id)
    if not _is_strict_int(observation_id) or observation_id < 0:
        return _invalid_resolution('invalid_observation_id', candidate_id)
    if not isinstance(candidates, dict) or candidate_id not in candidates:
        return _invalid_resolution('unknown_candidate_id', candidate_id)
    if not isinstance(masks, dict):
        return _invalid_resolution('invalid_masks_store', candidate_id)
    if not isinstance(endpoint_text, str) or not endpoint_text.strip():
        return _invalid_resolution('invalid_endpoint_text', candidate_id)

    record = candidates.get(candidate_id)
    if not isinstance(record, dict):
        return _invalid_resolution('invalid_candidate_record', candidate_id)
    if record.get('candidate_id') != candidate_id:
        return _invalid_resolution('candidate_id_mismatch', candidate_id)

    rec_obs_id = record.get('observation_id')
    if not _is_strict_int(rec_obs_id) or rec_obs_id != observation_id:
        return _invalid_resolution('stale_or_mismatched_observation_id', candidate_id)
    if 'observation_id' in region and region.get('observation_id') is not None:
        if not _is_strict_int(region.get('observation_id')) or region.get('observation_id') != observation_id:
            return _invalid_resolution('region_observation_id_mismatch', candidate_id)

    view = record.get('view')
    if not isinstance(view, str) or view not in VALID_VIEWS:
        return _invalid_resolution('invalid_candidate_view', candidate_id)
    if not _valid_candidate_id(candidate_id, observation_id, view):
        return _invalid_resolution('malformed_candidate_id', candidate_id)
    if region.get('view') is not None and region.get('view') != view:
        return _invalid_resolution('region_view_mismatch', candidate_id)

    cand_entity_id = record.get('entity_id')
    if cand_entity_id is not None and (not _is_strict_int(cand_entity_id) or cand_entity_id < 0):
        return _invalid_resolution('invalid_candidate_entity_id', candidate_id)
    if 'entity_id' in region and region.get('entity_id') is not None:
        req_entity_id = region.get('entity_id')
        if not _is_strict_int(req_entity_id) or req_entity_id < 0 or cand_entity_id is None or req_entity_id != cand_entity_id:
            return _invalid_resolution('entity_id_conflict', candidate_id)

    conf = record.get('confidence')
    if not _is_finite_float(conf) or not (0.0 <= float(conf) <= 1.0):
        return _invalid_resolution('invalid_candidate_confidence', candidate_id)

    det_name = record.get('detector_class_name')
    canon_name = record.get('canonical_label')
    labels = [lbl.strip() for lbl in (det_name, canon_name) if isinstance(lbl, str) and lbl.strip()]
    if not labels:
        return _invalid_resolution('missing_candidate_label', candidate_id)
    if (isinstance(det_name, str) and det_name.strip() and isinstance(canon_name, str) and canon_name.strip()
            and semantic_evidence.canonical_label(det_name) != semantic_evidence.canonical_label(canon_name)):
        return _invalid_resolution('conflicting_candidate_labels', candidate_id)
    if not all(target_identity.detector_matches_endpoint(lbl, endpoint_text, role=evidence_role) for lbl in labels):
        return _invalid_resolution('class_mismatch_for_endpoint_role', candidate_id)

    size = _parse_image_size(record.get('image_size'))
    if size is None:
        return _invalid_resolution('invalid_image_size', candidate_id)
    width, height = size

    cand_bbox = _parse_norm_bbox(record.get('bbox'))
    if cand_bbox is None:
        return _invalid_resolution('invalid_candidate_bbox', candidate_id)

    mask_key = record.get('mask_key')
    if not isinstance(mask_key, str) or not mask_key.strip():
        return _invalid_resolution('missing_candidate_mask', candidate_id)
    if mask_key != candidate_id:
        return _invalid_resolution('candidate_mask_key_mismatch', candidate_id)
    if mask_key not in masks:
        return _invalid_resolution('missing_candidate_mask', candidate_id)
    raw_mask = masks.get(mask_key)
    if raw_mask is None:
        return _invalid_resolution('null_candidate_mask', candidate_id)
    mask_arr = np.asarray(raw_mask)
    if mask_arr.ndim != 2 or mask_arr.shape != (height, width):
        return _invalid_resolution('mask_shape_mismatch', candidate_id)
    mask_bool = np.asarray(mask_arr > 0, dtype=bool)
    total_mask_px = int(np.count_nonzero(mask_bool))
    if total_mask_px == 0:
        return _invalid_resolution('empty_candidate_mask', candidate_id)

    cpx1, cpy1, cpx2, cpy2 = _bbox_to_px(cand_bbox, width, height)
    if cpx2 <= cpx1 or cpy2 <= cpy1 or not np.any(mask_bool[cpy1:cpy2, cpx1:cpx2]):
        return _invalid_resolution('candidate_mask_bbox_disjoint', candidate_id)
    bpx1, bpy1 = max(0, cpx1 - 1), max(0, cpy1 - 1)
    bpx2, bpy2 = min(width, cpx2 + 1), min(height, cpy2 + 1)
    if int(np.count_nonzero(mask_bool[bpy1:bpy2, bpx1:bpx2])) != total_mask_px:
        return _invalid_resolution('mask_outside_candidate_bbox', candidate_id)

    req_bbox = region.get('bbox')
    bbox_space = region.get('bbox_space')
    if req_bbox is None:
        if bbox_space is not None and bbox_space != 'full_view':
            return _invalid_resolution('invalid_bbox_space', candidate_id)
        selected_bbox = list(cand_bbox)
        selected_mask = mask_bool.copy()
    else:
        if bbox_space != 'full_view':
            return _invalid_resolution('invalid_or_missing_bbox_space', candidate_id)
        parsed_req_bbox = _parse_norm_bbox(req_bbox)
        if parsed_req_bbox is None:
            return _invalid_resolution('invalid_region_bbox', candidate_id)
        ix1 = max(cand_bbox[0], parsed_req_bbox[0])
        iy1 = max(cand_bbox[1], parsed_req_bbox[1])
        ix2 = min(cand_bbox[2], parsed_req_bbox[2])
        iy2 = min(cand_bbox[3], parsed_req_bbox[3])
        if ix2 <= ix1 or iy2 <= iy1:
            return _invalid_resolution('region_bbox_outside_candidate_bbox', candidate_id)
        rpx1, rpy1, rpx2, rpy2 = _bbox_to_px(parsed_req_bbox, width, height)
        if rpx2 <= rpx1 or rpy2 <= rpy1 or not np.any(mask_bool[rpy1:rpy2, rpx1:rpx2]):
            return _invalid_resolution('region_bbox_no_mask_overlap', candidate_id)
        selected_bbox = parsed_req_bbox
        selected_mask = np.zeros_like(mask_bool)
        selected_mask[rpy1:rpy2, rpx1:rpx2] = mask_bool[rpy1:rpy2, rpx1:rpx2]

    chosen_det = str(det_name).strip() if isinstance(det_name, str) and det_name.strip() else labels[0]
    chosen_canon = semantic_evidence.canonical_label(canon_name if isinstance(canon_name, str) and canon_name.strip() else chosen_det)
    return {
        'valid': True,
        'reason': 'ok',
        'candidate_id': candidate_id,
        'entity_id': cand_entity_id,
        'view': view,
        'bbox': selected_bbox,
        'bbox_space': 'full_view',
        'mask': selected_mask,
        'mask_key': mask_key,
        'mask_view': view,
        'mask_observation_id': observation_id,
        'image_size': [width, height],
        'confidence': float(conf),
        'selected_detector_class_name': chosen_det,
        'canonical_label': chosen_canon,
    }


def _extract_raw_rgb(raw_image):
    if isinstance(raw_image, Image.Image):
        if raw_image.mode not in ('RGB', 'RGBA'):
            return None
        w, h = raw_image.size
        return (raw_image, w, h) if w > 0 and h > 0 else None
    if isinstance(raw_image, np.ndarray):
        if raw_image.ndim != 3 or raw_image.shape[2] not in (3, 4) or raw_image.dtype != np.uint8:
            return None
        h, w = int(raw_image.shape[0]), int(raw_image.shape[1])
        return (raw_image, w, h) if w > 0 and h > 0 else None
    return None


def _normalize_roles(roles_map, candidate_id):
    if not isinstance(roles_map, dict) or candidate_id not in roles_map:
        return []
    raw = roles_map.get(candidate_id)
    if isinstance(raw, str):
        raw = [raw]
    elif not isinstance(raw, (list, tuple, set)):
        return []
    out = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            role = item.strip().lower()
            if role not in out:
                out.append(role)
    return out


def select_candidate_crops(candidates, rgb_views, *, observation_id, roles=None, limit=4, max_side=640, padding=0.15):
    """Select and crop current-step high-resolution RGB candidate views without mutating inputs."""
    if not _is_strict_int(observation_id) or observation_id < 0:
        return []
    if not isinstance(rgb_views, dict) or not _is_strict_int(limit) or limit <= 0:
        return []
    if not _is_strict_int(max_side) or max_side <= 0:
        return []
    if not _is_finite_float(padding) or not (0.0 <= float(padding) <= 1e6):
        return []

    bounded_limit = min(int(limit), _MAX_CROP_LIMIT)
    pad_ratio = float(padding)

    if isinstance(candidates, dict):
        items = list(candidates.items())
    elif isinstance(candidates, (list, tuple)):
        items = [(r.get('candidate_id') if isinstance(r, dict) else None, r) for r in candidates]
    else:
        return []

    valid_entries = []
    seen_ids = set()
    for key_id, record in items:
        if not isinstance(record, dict):
            continue
        cid = key_id if isinstance(key_id, str) else record.get('candidate_id')
        if not isinstance(cid, str) or cid in seen_ids:
            continue
        if record.get('candidate_id') != cid:
            continue
        rec_obs = record.get('observation_id')
        if not _is_strict_int(rec_obs) or rec_obs != observation_id:
            continue
        view = record.get('view')
        if not isinstance(view, str) or view not in VALID_VIEWS or view not in rgb_views:
            continue
        if not _valid_candidate_id(cid, observation_id, view):
            continue
        raw_info = _extract_raw_rgb(rgb_views.get(view))
        if raw_info is None:
            continue
        raw_img, img_w, img_h = raw_info
        size = _parse_image_size(record.get('image_size'))
        if size is None or size != (img_w, img_h):
            continue
        bbox = _parse_norm_bbox(record.get('bbox'))
        if bbox is None:
            continue
        conf = record.get('confidence')
        if not _is_finite_float(conf) or not (0.0 <= float(conf) <= 1.0):
            continue
        det_name = record.get('detector_class_name')
        canon_name = record.get('canonical_label')
        label_str = str(det_name or canon_name or '').strip()
        if not label_str:
            continue

        seen_ids.add(cid)
        cand_roles = _normalize_roles(roles, cid)
        best_role_rank = min((_ROLE_PRIORITY.get(r, 4) for r in cand_roles), default=5)
        sort_key = (0 if cand_roles else 1, best_role_rank, -float(conf), VALID_VIEWS.index(view), cid)
        valid_entries.append((sort_key, cid, record, raw_img, img_w, img_h, bbox, float(conf), label_str, cand_roles))

    valid_entries.sort(key=lambda item: item[0])
    results = []
    resample = getattr(Image, 'Resampling', Image).BILINEAR

    for _, cid, record, raw_img, width, height, bbox, conf, label_str, cand_roles in valid_entries[:bounded_limit]:
        view = record.get('view')
        x1, y1, x2, y2 = bbox
        bw = (x2 - x1) * width
        bh = (y2 - y1) * height
        pad_x = bw * pad_ratio
        pad_y = bh * pad_ratio
        left = max(0, min(width - 1, int(math.floor(x1 * width - pad_x + 1e-9))))
        top = max(0, min(height - 1, int(math.floor(y1 * height - pad_y + 1e-9))))
        right = max(left + 1, min(width, int(math.ceil(x2 * width + pad_x - 1e-9))))
        bottom = max(top + 1, min(height, int(math.ceil(y2 * height + pad_y - 1e-9))))

        if isinstance(raw_img, np.ndarray):
            sub = np.ascontiguousarray(raw_img[top:bottom, left:right, :3])
            pil_crop = Image.fromarray(sub, mode='RGB')
        else:
            pil_crop = raw_img.crop((left, top, right, bottom)).convert('RGB')

        cw, ch = pil_crop.size
        if max(cw, ch) > max_side:
            if cw >= ch:
                new_w = max_side
                new_h = max(1, int(round(ch * (max_side / float(cw)))))
            else:
                new_h = max_side
                new_w = max(1, int(round(cw * (max_side / float(ch)))))
            pil_crop = pil_crop.resize((new_w, new_h), resample=resample)

        canon = semantic_evidence.canonical_label(record.get('canonical_label') or label_str)
        role_tag = ','.join(cand_roles) if cand_roles else 'context'
        caption = (
            f"candidate_id={cid} observation_id={observation_id} view={view} class={canon} conf={conf:.2f} roles={role_tag} "
            f"original_bbox=[{x1:.4f},{y1:.4f},{x2:.4f},{y2:.4f}] bbox_space=FULL_VIEW "
            f"(all bbox outputs must use FULL_VIEW coordinates only, never crop coordinates)"
        )
        results.append({
            'image': pil_crop,
            'candidate_id': cid,
            'observation_id': int(observation_id),
            'view': view,
            'detector_class_name': str(record.get('detector_class_name') or label_str),
            'canonical_label': canon,
            'confidence': conf,
            'original_bbox': list(bbox),
            'bbox_space': 'full_view',
            'crop_bounds_px': [int(left), int(top), int(right), int(bottom)],
            'crop_size': [int(pil_crop.size[0]), int(pil_crop.size[1])],
            'image_size': [int(width), int(height)],
            'roles': list(cand_roles),
            'caption': caption,
        })
    return results

