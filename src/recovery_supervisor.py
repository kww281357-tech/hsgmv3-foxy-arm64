"""Physical recovery evidence from executed motion, independent of model verdicts.

This module selects no semantic goal and never moves the simulator. Its bounded
windows survive waypoint replans, so repeated tiny motion cannot reset a stall.
"""
from collections import deque
import math

import numpy as np


def finite_pose(value):
    try:
        point = np.asarray(value, dtype=float)
        return point if point.shape == (3,) and np.isfinite(point).all() else None
    except (TypeError, ValueError):
        return None


class RecoverySupervisor:
    def __init__(self, cfg=None):
        self.cfg = cfg or {}
        self.window = max(8, int(self.cfg.get('physical_recovery_window_steps', 16)))
        self.rows = deque(maxlen=self.window)
        self.consumed = 0
        self.key = None
        self.hold_origin = None
        self.held_records = 0

    def reset_window(self):
        self.rows.clear()

    def consume(self, ledger, key, confirming=False):
        """Consume each real ledger row once; return physical evidence or None.

        Confirmation exempts at most four observation records at one place.
        Toggling its flag or changing waypoint IDs cannot replenish this grant.
        Only measured displacement of .35 m starts a new observation location.
        """
        if len(ledger) < self.consumed:
            return {'reason': 'physical_ledger_rewound', 'abort': True}
        if key != self.key:
            self.rows.clear()
            self.key = key
            self.hold_origin = None
            self.held_records = 0
        new = ledger[self.consumed:]
        self.consumed = len(ledger)
        for row in new:
            if row.get('subtask_id') != key:
                self.rows.clear()
                continue
            if row.get('recovery'):
                self.rows.clear()
                continue
            before, after = finite_pose(row.get('pose_before')), finite_pose(row.get('pose'))
            if before is None or after is None:
                return {'reason': 'physical_localization_invalid', 'abort': True}
            distance = float(np.linalg.norm(after - before))
            # Production _action_to_text(null) is intentionally empty. It is
            # still a measured zero-motion tick, including endpoint handbacks.
            null_text = (row.get('action_type') == 'null' and row.get('executed_action') == ''
                         and distance <= 1e-6 and row.get('requested_distance') == 0)
            if (type(row.get('step')) is not int or not isinstance(row.get('executed_action'), str)
                    or not (row['executed_action'] or null_text)):
                return {'reason': 'physical_execution_record_invalid', 'abort': True}
            if self.rows and (row['step'] != self.rows[-1]['step'] + 1
                              or np.linalg.norm(before - self.rows[-1]['after']) > .05):
                self.rows.clear()  # Unknown gaps are never evidence of a continuous loop.
            if self.hold_origin is None or np.linalg.norm(after - self.hold_origin) >= .35:
                self.hold_origin = after.copy()
                self.held_records = 0
            hold_limit = min(6, max(0, int(self.cfg.get('physical_recovery_hold_records', 4))))
            if (confirming and row.get('action_type') in ('turn', 'null')
                    and distance < .02 and self.held_records < hold_limit):
                self.held_records += 1
                self.rows.clear()
                continue
            try:
                requested = float(row.get('requested_distance', 0))
            except (TypeError, ValueError):
                requested = math.nan
            if not math.isfinite(requested):
                return {'reason': 'physical_execution_record_invalid', 'abort': True}
            self.rows.append({'step': row['step'], 'before': before, 'after': after,
                              'distance': distance, 'kind': row.get('action_type'),
                              'blocked': row.get('action_type') == 'move_forward'
                                  and requested >= .08 and distance < .02,
                              'record': row})
        if not new or not self.rows:
            return None
        rows = list(self.rows)
        recent = rows[-6:]
        blocked = [r for r in recent if r['blocked']]
        recent_net = (float(np.linalg.norm(recent[-1]['after'] - blocked[0]['before']))
                      if blocked else math.inf)
        if len(blocked) >= 3 and recent_net <= .15:
            return self._evidence('blocked_translation', blocked,
                                  [r for r in recent if r['step'] >= blocked[0]['step']])
        if len(rows) < self.window:
            return None
        points = np.array([rows[0]['before'], *(r['after'] for r in rows)])
        extent = float(np.linalg.norm(points - points[0], axis=1).max())
        path = sum(r['distance'] for r in rows)
        net = float(np.linalg.norm(points[-1] - points[0]))
        attempts = sum(r['kind'] == 'move_forward' for r in rows)
        turns = sum(r['kind'] == 'turn' for r in rows)
        waits = sum(r['kind'] == 'null' for r in rows)
        if extent < .18 and (attempts >= 4 or turns >= 6 or waits >= 8):
            return self._evidence('stationary_execution', rows, rows)
        if path >= 3. and net <= .45 and net / path < .2:
            return self._evidence('measured_revisit_loop', rows, rows)
        return None

    @staticmethod
    def _evidence(reason, affected, window):
        result = {'reason': reason, 'first_affected_step': affected[0]['step'],
                  'last_affected_step': affected[-1]['step'],
                  'steps': [r['step'] for r in affected],
                  'window_path_m': sum(r['distance'] for r in window),
                  'window_net_m': float(np.linalg.norm(window[-1]['after'] - window[0]['before'])),
                  'blocked_attempts': sum(r['blocked'] for r in window)}
        # Only a real executed waypoint can identify a temporarily failed target.
        for item in reversed(window):
            row = item['record']
            packet, requested = row.get('waypoint_execution') or {}, row.get('requested_action') or {}
            if not isinstance(packet, dict) or not isinstance(requested, dict):
                continue
            target = finite_pose(packet.get('target'))
            if (target is not None and requested.get('type') == 'navigate'
                    and requested.get('waypoint_id') and requested['waypoint_id'] == packet.get('action_id')):
                result.update(failed_target=target.tolist(), failed_target_step=item['step'])
                break
        return result


def witnessed_checkpoint(checkpoint, ledger, key, floor, before_step):
    """A safe-anchor candidate must be a saved boundary on the actual trajectory.

    ``floor`` is diagnostic metadata: mapper labels may change on stair treads.
    The caller checks measured height and a fresh connected NavMesh return.
    """
    if not isinstance(checkpoint, dict):
        return False
    index = checkpoint.get('ledger_len')
    point = finite_pose(checkpoint.get('ground_pose'))
    if (checkpoint.get('subtask_key') != key
            or type(index) is not int or not 0 <= index <= before_step
            or index >= len(ledger) or point is None):
        return False
    record = ledger[index]
    witness = finite_pose(record.get('pose_before'))
    return bool(record.get('step') == index and record.get('subtask_id') == key
                and not record.get('recovery') and not record.get('rejected_by_rollback') and witness is not None
                and np.linalg.norm(point - witness) <= .02)

