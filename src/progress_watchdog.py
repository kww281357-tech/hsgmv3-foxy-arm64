"""Evidence-aware bounded recovery. Never consumes evaluator goal coordinates."""
from collections import deque
import math
import numpy as np
from runtime_budget import EpisodeBudgetExceeded


class NavigationAborted(EpisodeBudgetExceeded):
    """Unfinished navigation, distinct from task STOP and infrastructure failure."""


def supported_spatial_rejection(result):
    return bool(result.get('available') and not result.get('outcome')
                and not result.get('contract_conflict')
                and result.get('evidence_state') == 'contradicted'
                and result.get('confidence') in ('medium', 'high'))


def witnessed_progress(subtasks):
    """Stable IDs prevent resetting the watchdog by repeatedly re-verifying old evidence."""
    tokens = set()
    for key, st in subtasks.items():
        if st.get('completed') and st.get('completion_evidence_reliable'):
            tokens.add(str(key) + ':completed')
        for check in st.get('completion_conditions', []):
            if check.get('status') == 'satisfied' and check.get('execution_checked'):
                tokens.add(str(key) + ':' + str(check.get('id')))
    return tokens


class ProgressWatchdog:
    def __init__(self, cfg=None):
        self.cfg = cfg or {}
        self.window = max(8, int(self.cfg.get('progress_window_steps', 16)))
        self.samples = deque(maxlen=self.window)
        self.seen_cells = set()
        self.achieved = set()
        self.last_progress_step = 0
        self.last_progress_path = 0.
        self.last_step = -1
        self.last_pose = None
        self.path = 0.
        self.interventions = 0
        self.segment_interventions = 0
        self.recovery_end_step = None
        self.last_intervention_step = -10000

    def observe(self, step, pose, progress=(), recovering=False, confirming=False):
        if step <= self.last_step:
            return None
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            return {'action': 'abort', 'reason': 'invalid_localization'}
        if self.last_pose is not None:
            self.path += float(np.linalg.norm(pose-self.last_pose))
        self.last_pose = pose.copy()
        self.last_step = int(step)
        progress = set(progress)
        if progress - self.achieved:
            self.achieved |= progress
            self.last_progress_step = step
            self.last_progress_path = self.path
            self.segment_interventions = 0
            self.samples.clear()
        # Height is included so stairs/floors cannot masquerade as horizontal loops.
        cell = tuple(math.floor(float(x)/.75) for x in pose)
        novel = cell not in self.seen_cells
        self.seen_cells.add(cell)
        self.samples.append((step, pose.copy(), novel))
        if recovering:
            self.recovery_end_step = step
            return None  # Recovery has its own strict tick/distance/attempt budget.
        if confirming:
            return None  # Terminal hold has priority and its own bounded observations.
        grace = max(1, int(self.cfg.get('progress_recovery_grace_steps', 12)))
        anchor = max(self.last_intervention_step, self.recovery_end_step or -10000)
        if step-anchor < grace or len(self.samples) < self.window:
            return None
        if step-self.last_progress_step < self.window:
            return None
        points = np.array([row[1] for row in self.samples])
        path = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        net = float(np.linalg.norm(points[-1]-points[0]))
        radius = float(np.linalg.norm(points-points[0], axis=1).max())
        new_fraction = sum(row[2] for row in self.samples)/len(self.samples)
        stagnant = radius < .45
        looping = path >= 3 and net/max(path, 1e-6) < .2 and new_fraction <= .15
        # Fresh scenery alone cannot buy unlimited time without instruction progress.
        drought = (step-self.last_progress_step >= int(self.cfg.get('progress_max_unverified_steps', 48))
                   and self.path-self.last_progress_path >= float(self.cfg.get('progress_max_unverified_path_m', 18)))
        if not (stagnant or looping or drought):
            return None
        reason = 'stationary' if stagnant else ('revisit_loop' if looping else 'unverified_travel')
        decision = {'reason': reason, 'window_path_m': round(path, 3), 'window_net_m': round(net, 3),
                    'novel_fraction': round(new_fraction, 3),
                    'steps_since_progress': step-self.last_progress_step}
        if (self.segment_interventions >= int(self.cfg.get('progress_recoveries_without_progress', 1))
                or self.interventions >= int(self.cfg.get('progress_max_interventions', 3))):
            return {'action': 'abort', **decision}
        self.interventions += 1
        self.segment_interventions += 1
        self.last_intervention_step = step
        self.samples.clear()
        return {'action': 'recover', **decision}

