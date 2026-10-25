"""Small perception contracts shared by mapping and endpoint verification.

No detector or simulator imports. Image coordinates always refer to the original
RGB/depth image; bounding boxes are normalized xyxy with exclusive right/bottom.
"""
import re

import numpy as np


def canonical_label(name):
    label = re.sub(r'[\s_-]+', '_', str(name).strip().lower())
    return {
        'stair': 'stairs', 'staircase': 'stairs', 'staircases': 'stairs',
        'sculpture': 'statue', 'sculptures': 'statue', 'statues': 'statue',
        'eye_charts': 'eye_chart', 'eye_test_chart': 'eye_chart',
        'eye_test_charts': 'eye_chart', 'snellen_chart': 'eye_chart',
        'snellen_charts': 'eye_chart',
    }.get(label, label)


def rgb_to_bgr(image):
    """Ultralytics NumPy input is BGR, whereas simulator images are RGB(A)."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] not in (3, 4) or image.dtype != np.uint8:
        raise ValueError('Expected an RGB or RGBA uint8 image.')
    return np.ascontiguousarray(image[:, :, :3][:, :, ::-1])


def detection_observation(mask, depth, confidence, *, observation_id, step, view, mask_key):
    mask = np.asarray(mask) > 0
    depth = np.asarray(depth)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if mask.ndim != 2 or depth.shape != mask.shape:
        raise ValueError('Detection mask and depth must share the original image shape.')
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    valid = mask & np.isfinite(depth) & (depth > 0)
    values = depth[valid]
    return {
        'observation_id': int(observation_id), 'step': step, 'view': str(view),
        'confidence': float(confidence), 'mask_key': str(mask_key),
        'bbox': [float(xs.min() / width), float(ys.min() / height),
                 float((xs.max() + 1) / width), float((ys.max() + 1) / height)],
        'image_size': [int(width), int(height)], 'mask_pixels': int(len(xs)),
        'touches_image_edge': bool(xs.min() == 0 or ys.min() == 0
                                   or xs.max() == width - 1 or ys.max() == height - 1),
        'valid_depth_count': int(len(values)),
        'valid_depth_fraction': float(len(values) / len(xs)),
        'depth_p10_m': float(np.quantile(values, .1)) if len(values) else None,
        'depth_median_m': float(np.median(values)) if len(values) else None,
    }


def add_entity_observation(entity, observation):
    """Keep current-view evidence distinct from an accumulated object cloud."""
    revision = observation['observation_id']
    same_observation = entity.get('last_seen_observation_id') == revision
    previous = entity.get('observations', []) if same_observation else []
    # One best detection per view; multiple views do not imply temporal support.
    by_view = {row['view']: row for row in previous}
    view = observation['view']
    if view not in by_view or observation['confidence'] >= by_view[view]['confidence']:
        by_view[view] = dict(observation)
    rows = list(by_view.values())
    entity.setdefault('first_seen_step', observation['step'])
    entity.setdefault('first_seen_observation_id', revision)
    entity['last_seen_step'] = observation['step']
    entity['last_seen_observation_id'] = revision
    entity['observation_count'] = int(entity.get('observation_count', 0)) + int(not same_observation)
    entity['detection_count'] = int(entity.get('detection_count', 0)) + 1
    entity['observations'] = rows
    entity['last_seen_views'] = sorted(by_view)
    entity['confidence'] = max(row['confidence'] for row in rows)
    entity['max_confidence'] = max(float(entity.get('max_confidence', 0)), entity['confidence'])
    return entity


def merge_entity_evidence(entity, observed):
    if canonical_label(entity.get('class_name', '')) != canonical_label(observed.get('class_name', '')):
        raise ValueError('A new detector class cannot relabel an existing object identity.')
    for observation in observed.get('observations', []):
        add_entity_observation(entity, observation)
    return entity


def entity_observed_now(entity, observation_id):
    return (observation_id is not None and entity.get('last_seen_observation_id') == observation_id
            and any(row.get('observation_id') == observation_id for row in entity.get('observations', [])))

