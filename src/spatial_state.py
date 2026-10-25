"""Measured spatial memory and cheap perception scheduling; no goal-truth inputs."""
import math
import re
import numpy as np


BEV_OVERLAY_LEGEND = (
    'BEV legend: the translucent cyan/light-blue sector is a synthetic camera field-of-view overlay, '
    'not a rug, floor material, room boundary, or target. Map labels are detector predictions; '
    'region colors alone do not identify materials. Confirm identity and appearance in the current RGB views. '
    'Numbered circles are candidate movement waypoints, not the instruction endpoint. '
    'Use the map for geometry and recorded history; map annotations alone do not prove arrival.'
)


def relative_position(pose, rotation, target):
    """Use camera basis directly: mapper swaps Habitat axes, so yaw signs are unsafe."""
    delta = np.asarray(target, dtype=float) - np.asarray(pose, dtype=float)
    rotation = np.asarray(rotation, dtype=float)
    forward, right = -rotation[:2, 2], rotation[:2, 0]
    forward = forward / max(float(np.linalg.norm(forward)), 1e-8)
    right = right / max(float(np.linalg.norm(right)), 1e-8)
    f, r = float(delta[:2] @ forward), float(delta[:2] @ right)
    return {'distance_m': round(float(np.linalg.norm(delta[:2])), 2),
            'forward_m': round(f, 2), 'right_m': round(r, 2),
            'height_delta_m': round(float(delta[2]), 2)}


class SpatialState:
    """Bounded graph of physically visited places, not inferred room connectivity."""
    def __init__(self, radius=.75, capacity=128):
        self.radius = radius
        self.capacity = capacity
        self.nodes = {}
        self.edges = set()
        self.current = None
        self.next_id = 0
        self.recent = []
        self.distance = 0.

    def update(self, pose, step):
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            return
        if self.recent:
            self.distance += float(np.linalg.norm(pose - self.recent[-1]))
        self.recent = (self.recent + [pose.copy()])[-16:]
        neighbors = [(float(np.linalg.norm(pose - node['pose'])), ident)
                     for ident, node in self.nodes.items()]
        nearest = min(neighbors, default=(math.inf, None))
        ident = nearest[1] if nearest[0] <= self.radius else None
        if ident is None:
            ident = self.next_id
            self.next_id += 1
            self.nodes[ident] = {'pose': pose.copy(), 'entries': 0, 'last_step': step}
        if ident != self.current:
            self.nodes[ident]['entries'] += 1
            if self.current in self.nodes:
                self.edges.add((self.current, ident))
        self.nodes[ident]['last_step'] = int(step)
        self.current = ident
        while len(self.nodes) > self.capacity:
            victim = min((i for i in self.nodes if i != ident), key=lambda i: self.nodes[i]['last_step'])
            del self.nodes[victim]
            self.edges = {edge for edge in self.edges if victim not in edge}

    def packet(self, pose, rotation, actions, entities, instruction, limit=10):
        pose = np.asarray(pose, dtype=float)
        words = set(re.findall(r'[a-z]+', instruction.lower()))
        landmarks = []
        for ident, entity in enumerate(entities):
            center = np.asarray(entity.get('center', []), dtype=float)
            if center.shape != (3,) or not np.isfinite(center).all():
                continue
            name = str(entity.get('class_name', 'unknown'))
            rel = relative_position(pose, rotation, center)
            relevant = bool(set(re.findall(r'[a-z]+', name.lower())) & words)
            if rel['distance_m'] <= 8 and abs(rel['height_delta_m']) <= 3:
                landmarks.append((not relevant, rel['distance_m'], ident, name, center, rel))
        landmarks.sort(key=lambda x: x[:3])
        landmarks = landmarks[:6]
        candidates = []
        for ident, action in actions.items():
            if action.get('type') != 'waypoint':
                continue
            target = np.asarray(action.get('target_point_world', []), dtype=float)
            if target.shape != (3,) or not np.isfinite(target).all():
                continue
            # Places store camera height; candidate waypoints are ground points.
            visited = any(np.linalg.norm(target[:2] - n['pose'][:2]) <= self.radius
                          and abs(target[2] - n['pose'][2]) <= 1.6 for n in self.nodes.values())
            gains = {str(row[2]): round(float(np.linalg.norm(row[4][:2]-pose[:2]) -
                                                  np.linalg.norm(row[4][:2]-target[:2])), 2)
                     for row in landmarks if not row[0]}
            candidates.append({'action': str(ident), **relative_position(pose, rotation, target),
                               'visited_place': visited, 'rejected': bool(action.get('rejected')),
                               'landmark_approach_m': gains})
        nearby = sorted(self.nodes, key=lambda i: np.linalg.norm(self.nodes[i]['pose'] - pose))[:6]
        return {'frame': 'meters; +forward=ahead, +right=right; map landmarks are historical, not current visibility',
                'place': self.current, 'path_m': round(self.distance, 2),
                'recent_net_m': round(float(np.linalg.norm(pose-self.recent[0])), 2) if self.recent else 0,
                'places': [{'id': i, 'entries': self.nodes[i]['entries'],
                            **relative_position(pose, rotation, self.nodes[i]['pose'])} for i in nearby],
                'traversed_edges': [list(e) for e in sorted(self.edges) if e[0] in nearby and e[1] in nearby][-10:],
                'landmarks': [{'map_id': row[2], 'name': row[3], 'instruction_match': not row[0], **row[5]}
                              for row in landmarks], 'candidates': candidates[:limit],
                'candidates_omitted': max(0, len(candidates)-limit),
                'rule': 'Approach is geometric, not proof of traversability or completion. Revisit only when the instruction/recovery requires it.'}


def semantic_refresh_due(step, last_step, pose, last_pose, yaw, last_yaw, *, critical=False,
                         max_interval=3, displacement=.6, turn_radians=math.pi/4):
    if critical or last_pose is None or step-last_step >= max_interval:
        return True
    moved = np.linalg.norm(np.asarray(pose)-np.asarray(last_pose))
    turn = abs((yaw-last_yaw+math.pi) % (2*math.pi)-math.pi)
    return bool(moved >= displacement or turn >= turn_radians)


def planner_image_content(encode, front, left, right, bev, history, compact=False):
    """Current front and BEV always survive; verifier does not use this filter."""
    selected = [('earlier executed front view', image) for image in (history[-1:] if compact else history[-3:])]
    if not compact:
        selected += [('current left view', left), ('current right view', right)]
    selected += [('current front view', front), ('current heading-up BEV', bev)]
    content = []
    for label, image in selected:
        if label == 'current heading-up BEV':
            label += '. ' + BEV_OVERLAY_LEGEND
        content += [{'type': 'text', 'text': label},
                    {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + encode(image), 'detail': 'high'}}]
    return content

