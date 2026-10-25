"""Measured, bounded recovery to a selected navigation checkpoint.

The planner chooses the checkpoint and certifies its route. This mixin owns the
recovery lifecycle; ``execute_action`` remains the sole writer of motion records
and travelled distance. No logical position or measured trajectory is rewound.
"""

import copy
import math

import numpy as np


class CheckpointRecovery:
    def _recovery_ground_pose(self):
        return np.asarray(self.mapper.current_position, dtype=float) + np.array(
            [0.0, 0.0, float(self.mapper.initial_floor_height)])

    def _recovery_error(self, pending):
        if not pending.get('checkpoint_valid', True):
            return dict.fromkeys(('position_error_m', 'horizontal_error_m', 'height_error_m', 'heading_error_deg'))
        actual = self._recovery_ground_pose()
        delta = actual - pending['target']
        heading = abs(self._normalize_angle(pending['target_yaw'] - self._get_agent_yaw()))
        return {'position_error_m': float(np.linalg.norm(delta)),
                'horizontal_error_m': float(np.linalg.norm(delta[:2])),
                'height_error_m': abs(float(delta[2])),
                'heading_error_deg': float(math.degrees(heading))}

    def _recovery_arrived(self, pending):
        error = self._recovery_error(pending)
        cfg = self.navigation_cfg
        return bool(all(v is not None and math.isfinite(v) for v in error.values())
                    and error['horizontal_error_m'] <= float(cfg.get('rollback_position_tolerance', .3))
                    and error['height_error_m'] <= float(cfg.get('rollback_height_tolerance_m', .25))
                    and error['heading_error_deg'] <= float(cfg.get('rollback_heading_tolerance_deg', 5)))

    def _recovery_event(self, kind, pending, reason):
        physical = pending.get('recovery_source') == 'physical'
        return {'role': 'controller', 'event': ('physical_recovery_' if physical else 'wrong_path_rollback_') + kind, 'status': kind,
                'step': int(getattr(self, '_last_step_index', 0)),
                'subtask_id': pending['subtask_id'], 'reason': str(reason or ''),
                'checkpoint_id': pending['checkpoint_id'],
                'checkpoint_step': pending['checkpoint'].get('step'),
                'checkpoint_ledger_len': pending['checkpoint'].get('ledger_len'),
                'checkpoint_floor_label': pending['checkpoint'].get('floor_level'),
                'actual_floor_label': getattr(self.mapper, 'floor_level', None),
                'checkpoint_branch_target': copy.deepcopy(pending['checkpoint'].get('chosen_target')),
                'checkpoint_branch_evidence': copy.deepcopy(pending['checkpoint'].get('chosen_target_evidence')),
                'recovery_source': pending.get('recovery_source', 'manual'),
                'recovery_constraint': copy.deepcopy(pending.get('recovery_constraint')),
                'physical_evidence': copy.deepcopy(pending.get('physical_evidence')),
                'checkpoint_valid': pending.get('checkpoint_valid', True),
                'target_position': pending['target'].tolist(),
                'actual_ground_pose': self._recovery_ground_pose().tolist(),
                'actual_pose': np.asarray(self.mapper.current_position, dtype=float).tolist(),
                'ticks': pending['ticks'], 'attempt': pending.get('attempt', 0),
                **self._recovery_error(pending)}

    def _record_recovery_event(self, event):
        if not hasattr(self, 'navigation_trace'):
            self.navigation_trace = []
        self.navigation_trace.append(event)
        st = self.instruction_obj.get_subtask_by_key(event['subtask_id'])
        if st is not None:
            st.setdefault('rollback_events', []).append(copy.deepcopy(event))

    def _clear_recovery_motion(self):
        self._pending_recovery = None
        self.rolling_back = False
        self.current_path = []
        self.action_sequence = []
        self._need_restore_heading = False
        self._waypoint_execution = None
        self._active_rollback_checkpoint = None
        self._executing_verifier_correction = False
        self._requested_action = None
        self.target_world_position = None
        self._stair_progress = None
        self._observation_turns_without_motion = 0
        self._no_move_steps = 0
        self.instruction_obj.current_step_count = 0

    def _begin_checkpoint_recovery(self, path, target_rotation, checkpoint=None,
                                   reason=None, subtask_key=None, recovery_constraint=None,
                                   recovery_source='manual'):
        """Begin one certified return. Return False if invalid or over budget.

        ``path`` and ``ground_pose`` use the mapper's ground coordinate frame.
        A branch checkpoint may use the current fields (ledger_len, rotation,
        mapper_pose, subtask_key) or the earlier checkpoint equivalents.
        """
        if getattr(self, '_pending_recovery', None) is not None:
            return False  # Never replace an in-flight transaction silently.
        key = subtask_key or self.instruction_obj.get_current_subtask_key()
        cp = copy.deepcopy(checkpoint or {})
        cp_key = cp.get('subtask_key', cp.get('subtask_id', key))
        reason = str(reason or 'Return to the previous decision checkpoint.')
        failure = None
        checkpoint_valid = True
        try:
            from route_progress import checkpoint_precedes_contradiction
            if recovery_constraint is not None or recovery_source == 'wrong_path':
                if not checkpoint_precedes_contradiction(cp, recovery_constraint, key):
                    raise ValueError('checkpoint does not precede the contradicted motion while preserving the verified prefix')
                if (cp.get('floor_level') != getattr(self.mapper, 'floor_level', 0)
                        or recovery_constraint['max_checkpoint_ledger_len'] >= len(getattr(self, '_execution_ledger', []))):
                    raise ValueError('contradiction boundary is not executed or checkpoint floor changed')
            points = np.asarray(path, dtype=float)
            target = np.asarray(cp.get('ground_pose', cp.get('start_position', points[-1])), dtype=float)
            target_yaw = float(self._rot_to_yaw(np.asarray(target_rotation, dtype=float)))
            if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 2
                    or target.shape != (3,) or not np.isfinite(points).all()
                    or not np.isfinite(target).all() or not math.isfinite(target_yaw)):
                raise ValueError('non-finite or malformed checkpoint route')
            if np.linalg.norm(points[-1] - target) > .10:
                raise ValueError('route does not end at the selected checkpoint')
            if cp_key != key:
                raise ValueError('checkpoint belongs to another subtask')
            route_length = float(np.linalg.norm(points[0] - self._recovery_ground_pose())
                                 + np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            if route_length > float(self.navigation_cfg.get('rollback_max_path_m', 8)):
                failure = 'Recovery route exceeds rollback_max_path_m.'
        except (IndexError, TypeError, ValueError) as exc:
            checkpoint_valid = False
            target = self._recovery_ground_pose()
            target_yaw = float(self._get_agent_yaw())
            points = np.asarray([target, target])
            failure = 'Invalid recovery route: ' + str(exc)
        counts = getattr(self, '_rollback_counts', {})
        attempts = int(getattr(self, '_recovery_attempts_total', 0))
        cfg = self.navigation_cfg
        if (counts.get(key, 0) >= int(cfg.get('max_rollbacks_per_subtask', 2))
                or attempts >= int(cfg.get('rollback_max_total_attempts', 3))
                or int(getattr(self, '_recovery_steps_total', 0)) >= int(cfg.get('rollback_total_step_budget', 60))
                or int(cfg.get('rollback_max_steps', 40)) <= 0):
            failure = 'Recovery attempt or step budget exhausted.'
        pending = {'checkpoint': cp, 'subtask_id': key, 'target': target,
                   'recovery_source': recovery_source,
                   'physical_evidence': copy.deepcopy(cp.get('physical_evidence')),
                   'recovery_constraint': copy.deepcopy(recovery_constraint),
                   'checkpoint_valid': checkpoint_valid,
                   'target_yaw': target_yaw, 'reason': reason, 'ticks': 0,
                   'blocked_moves': 0, 'attempt': attempts + (not failure),
                   'checkpoint_id': cp.get('checkpoint_id', f"{key}:{cp.get('step', 'entry')}"),
                   'begin_ledger_len': len(getattr(self, '_execution_ledger', [])),
                   'last_observed_ledger_len': len(getattr(self, '_execution_ledger', [])),
                   'from_pose': np.asarray(self.mapper.current_position, dtype=float).tolist()}
        self._last_rollback_info = None
        self._last_recovery_context = None
        if failure:
            self._pending_recovery = pending
            self._finish_checkpoint_recovery(False, failure)
            return False
        counts[key] = counts.get(key, 0) + 1
        self._rollback_counts = counts
        self._recovery_attempts_total = attempts + 1
        cp['rollback_visits'] = int(cp.get('rollback_visits', 0)) + 1
        if isinstance(checkpoint, dict):
            checkpoint['rollback_visits'] = cp['rollback_visits']
        self._pending_recovery = pending
        self._active_rollback_checkpoint = cp
        self.current_path = [p.copy() for p in points[1:]]
        self.action_sequence = []
        self._waypoint_execution = None
        self._need_restore_heading = False
        self._executing_verifier_correction = False
        self._requested_action = None
        self.rolling_back = True
        self.target_world_position = target.copy()
        if recovery_source != 'physical':
            self._invalidate_rejected_branch(pending)
        else:
            for name in ('_pending_endpoint_confirmation', '_pending_endpoint_relocalization',
                         '_pending_final_stop_confirmation'):
                setattr(self, name, None)
            endpoint = getattr(self, '_endpoint_controller', None)
            if callable(getattr(endpoint, 'reset', None)):
                endpoint.reset()
        self._record_recovery_event(self._recovery_event('started', pending, reason))
        return True

    def _supervise_physical_progress(self, step):
        """Start a measured physical escape, or explicitly abort if unavailable.

        Call before ordinary planning and its stationary aborts. A True return
        grants this tick exclusively to ``_recovery_action``. Existing in-flight
        recovery is also True; this method never executes an action itself.
        """
        from progress_watchdog import NavigationAborted
        from recovery_supervisor import RecoverySupervisor, finite_pose, witnessed_checkpoint

        self._last_step_index = step
        if getattr(self, '_pending_recovery', None) is not None:
            return True
        key = self.instruction_obj.get_current_subtask_key()
        if key is None:
            return False
        supervisor = getattr(self, '_physical_recovery_supervisor', None)
        if supervisor is None:
            supervisor = self._physical_recovery_supervisor = RecoverySupervisor(self.navigation_cfg)
        ledger = getattr(self, '_execution_ledger', [])
        confirming = any(bool(getattr(self, name, None)) for name in (
            '_pending_endpoint_confirmation', '_pending_endpoint_relocalization',
            '_pending_final_stop_confirmation'))
        evidence = supervisor.consume(ledger, key, confirming=confirming)
        escape = getattr(self, '_endpoint_escape_requested', None)
        self._endpoint_escape_requested = None
        if isinstance(escape, dict) and escape.get('subtask_id') == key:
            boundary = escape.get('first_affected_step')
            if (escape.get('scope_key') == key + '.endpoint' and type(boundary) is int
                    and 0 <= boundary <= len(ledger)
                    and isinstance(escape.get('reason'), str) and escape['reason']):
                if not (evidence or {}).get('abort'):
                    evidence = {'reason': 'endpoint_observation_exhausted',
                        'endpoint_reason': escape['reason'], 'first_affected_step': boundary,
                        'last_affected_step': len(ledger) - 1,
                        'steps': [r['step'] for r in ledger[boundary:]
                                  if r.get('subtask_id') == key and not r.get('recovery')
                                  and type(r.get('step')) is int],
                        'source': 'bounded_endpoint_controller'}
            else:
                evidence = {'reason': 'physical_endpoint_escape_invalid', 'abort': True}
        retry = getattr(self, '_physical_recovery_retry', None)
        if retry and retry.get('subtask_id') == key and not (evidence or {}).get('abort'):
            evidence = retry['evidence']
        self._physical_recovery_retry = None
        if evidence is None:
            return False

        def abort(reason):
            self.navigation_trace.append({'role': 'controller', 'event': 'physical_recovery_unavailable',
                'status': 'aborted', 'step': step, 'subtask_id': key, 'recovery_source': 'physical',
                'reason': reason, 'physical_evidence': copy.deepcopy(evidence),
                'actual_ground_pose': self._recovery_ground_pose().tolist()})
            raise NavigationAborted(reason)

        if evidence.get('abort'):
            abort(evidence['reason'])
        counts = getattr(self, '_physical_recovery_counts', {})
        if counts.get(key, 0) >= min(2, max(0, int(self.navigation_cfg.get('physical_recovery_max_attempts', 2)))):
            abort('physical_recovery_attempt_budget_exhausted')
        current = self._recovery_ground_pose()
        floor = getattr(self.mapper, 'floor_level', 0)
        rejected = getattr(self, '_physical_recovery_failed_anchors', set())
        candidates = []
        for cp in getattr(self, '_branch_checkpoints', []):
            if not witnessed_checkpoint(cp, ledger, key, floor, evidence['first_affected_step']):
                continue
            point = finite_pose(cp.get('ground_pose'))
            identity = (key, cp['ledger_len'], tuple(point))
            try:
                rotation = np.asarray(cp.get('rotation'), dtype=float)
                yaw = float(self._rot_to_yaw(rotation))
            except (TypeError, ValueError, IndexError):
                continue
            if (identity in rejected or rotation.shape != (3, 3) or not np.isfinite(rotation).all()
                    or not math.isfinite(yaw) or int(cp.get('rollback_visits', 0)) >= 2):
                continue
            distance = float(np.linalg.norm(current[:2] - point[:2]))
            if .4 <= distance <= 4. and abs(float(current[2] - point[2])) <= .35:
                candidates.append((distance, cp, rotation, identity))
        candidates.sort(key=lambda item: item[0])
        for _, cp, rotation, identity in candidates:
            try:
                route = self._validated_waypoint_route(cp['ground_pose'], recovery=True)
                if not route:
                    continue
                path = np.asarray(route.get('path'), dtype=float)
                if (path.ndim != 2 or path.shape[1] != 3 or len(path) < 2
                        or not np.isfinite(path).all() or np.linalg.norm(path[-1] - cp['ground_pose']) > .1):
                    continue
                distance = float(np.linalg.norm(path[0] - current)
                                 + np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
                if distance > min(4., float(self.navigation_cfg.get('rollback_max_path_m', 8.))):
                    continue
            except (ValueError, TypeError, RuntimeError):
                continue  # An unknown NavMesh result is not a safe return certificate.
            # Attach evidence before begin so the started trace is complete.
            physical_cp = copy.deepcopy(cp)
            physical_cp['physical_evidence'] = copy.deepcopy(evidence)
            if not self._begin_checkpoint_recovery(path, rotation, physical_cp,
                    reason=evidence['reason'], subtask_key=key, recovery_source='physical'):
                abort('physical_recovery_transaction_unavailable')
            cp['rollback_visits'] = physical_cp['rollback_visits']
            counts[key] = counts.get(key, 0) + 1
            self._physical_recovery_counts = counts
            self._pending_recovery['physical_anchor_identity'] = identity
            supervisor.reset_window()
            return True
        abort('physical_recovery_no_safe_witnessed_anchor')

    def _remember_physical_failed_target(self, pending):
        """Short local target cooldown; physical trouble is not a semantic ban."""
        from recovery_supervisor import finite_pose
        evidence = pending.get('physical_evidence') or {}
        target = finite_pose(evidence.get('failed_target'))
        if target is None:
            return
        origin = np.asarray(self.mapper.current_position, dtype=float)
        radius = min(.25, .35 * float(np.linalg.norm(target[:2] - origin[:2])))
        if radius < .05:
            return
        now = int(getattr(self, '_last_step_index', 0))
        row = {'origin': origin.tolist(), 'target': target.tolist(), 'radius': radius,
               'expires': now + 12, 'source': 'physical_recovery',
               'evidence_step': evidence.get('failed_target_step')}
        self._blocked_targets = (getattr(self, '_blocked_targets', []) + [row])[-24:]

    def _invalidate_rejected_branch(self, pending):
        """Reject wrong-branch witnesses without erasing measured history."""
        cp, key = pending['checkpoint'], pending['subtask_id']
        cutoff = int(cp.get('ledger_len', cp.get('execution_ledger_length', 0)))
        end = pending['begin_ledger_len']
        rejected = set()
        st = self.instruction_obj.get_subtask_by_key(key)
        records = list(getattr(self, '_execution_ledger', [])) + list((st or {}).get('record', []))
        for row in records:
            step = row.get('step')
            if (row.get('subtask_id', key) == key and type(step) is int
                    and cutoff <= step < end and 'executed_action' in row):
                row['rejected_by_rollback'] = True
                rejected.add(step)
        if st is not None:
            st['completion_evidence_reliable'] = False
            for check in st.get('completion_conditions', []):
                if set(check.get('steps') or []) & rejected or cutoff == 0:
                    check.update(status='unknown', evidence='', steps=[], execution_checked=False)
        if cp.get('history_len') is not None and hasattr(self, 'history_msgs'):
            self.history_msgs = self.history_msgs[:max(0, int(cp['history_len']))]
        if not cp.get('is_subtask_entry'):
            self._remember_rejected_branch(pending)

    def _remember_rejected_branch(self, pending):
        """Remember an entry branch only from its recorded choice after arrival."""
        cp, key = pending['checkpoint'], pending['subtask_id']
        entry = cp.get('is_subtask_entry') is True
        if entry and (cp.get('chosen_target') is None or not cp.get('chosen_target_evidence')):
            return
        origin = np.asarray(cp.get('mapper_pose', pending['target'] - [0, 0, self.mapper.initial_floor_height]), dtype=float)
        wrong_target = cp.get('chosen_target', cp.get('target'))
        if wrong_target is None:
            wrong_target = self._recovery_ground_pose()
        wrong_target = np.asarray(wrong_target, dtype=float)
        now = int(getattr(self, '_last_step_index', 0))
        rows = getattr(self, '_blocked_targets', [])
        # A nearby initial waypoint must not blacklist every exit from entry.
        # Bound its point radius inside the existing 25-degree branch cone.
        radius = min(.8, .4 * float(np.linalg.norm(wrong_target[:2] - origin[:2]))) if entry else .8
        blocked = [{'origin': origin.tolist(), 'target': wrong_target.tolist(), 'radius': radius, 'expires': now + 30}]
        if not entry:
            blocked.append({'origin': origin.tolist(), 'target': self._recovery_ground_pose().tolist(),
                            'radius': .8, 'expires': now + 30})
        self._blocked_targets = (rows + blocked)[-24:]
        if hasattr(self, '_remember_failed_branch'):
            self._remember_failed_branch({
                'subtask_id': key, 'floor_level': cp.get('floor_level', getattr(self.mapper, 'floor_level', 0)),
                'start_position': pending['target'].tolist(), 'target': wrong_target.tolist(),
            }, pending['reason'], strong=True)

    def _finish_checkpoint_recovery(self, success, reason):
        """Commit completion only from the measured position and heading."""
        pending = getattr(self, '_pending_recovery', None)
        if pending is None:
            if getattr(self, 'rolling_back', False):
                self._clear_recovery_motion()
            return False
        if success and not self._recovery_arrived(pending):
            success = False
            reason = 'Checkpoint arrival was not physically confirmed: ' + str(reason)
        physical = pending.get('recovery_source') == 'physical'
        if success and pending['checkpoint'].get('is_subtask_entry') and not physical:
            self._remember_rejected_branch(pending)
        kind = 'completed' if success else 'failed'
        event = self._recovery_event(kind, pending, reason)
        self._record_recovery_event(event)
        current_pose = np.asarray(self.mapper.current_position, dtype=float).tolist()
        checkpoint_pose = (pending['target'] - [0, 0, self.mapper.initial_floor_height]).tolist()
        self._last_recovery_context = {
            'subtask_id': pending['subtask_id'], 'status': kind,
            'recovery_source': pending.get('recovery_source', 'manual'),
            'checkpoint_pose': checkpoint_pose, 'current_pose': current_pose,
            'reason': str(reason), **self._recovery_error(pending),
            'message': ('Physical return to the checkpoint was confirmed.' if success else
                        'Rollback failed or was cancelled; use the measured current pose, not the checkpoint.')}
        self._last_rollback_info = ({
            'subtask': pending['subtask_id'], 'status': 'completed',
            'from_pose': pending['from_pose'], 'to_pose': current_pose,
            'checkpoint_pose': checkpoint_pose,
            'blocked_view': pending['checkpoint'].get('chosen_view'),
            'reason': str(reason), 'shown_count': 0} if success and not physical else None)
        if physical:
            self._remember_physical_failed_target(pending)
            supervisor = getattr(self, '_physical_recovery_supervisor', None)
            if supervisor is not None:
                supervisor.reset_window()
            if success:
                self._physical_recovery_retry = None
            elif pending.get('physical_anchor_identity') is not None:
                failed = getattr(self, '_physical_recovery_failed_anchors', set())
                failed.add(pending['physical_anchor_identity'])
                self._physical_recovery_failed_anchors = failed
                self._physical_recovery_retry = {'subtask_id': pending['subtask_id'],
                    'evidence': copy.deepcopy(pending.get('physical_evidence'))}
        self._last_verifier_result = {}
        self._verifier_feedback = None
        self._last_arrival_check = {}
        self._uncertain_arrival_observation = {}
        self._clear_recovery_motion()
        return success

    def _cancel_checkpoint_recovery(self, reason):
        """Idempotent failure exit, including controller path invalidation."""
        return self._finish_checkpoint_recovery(False, reason)

    def _recovery_action(self, step):
        """Select one recovery action; a terminal tick returns a null action.

        The caller must bypass ordinary planning while pending, and must not
        replan during a terminal tick. Physical execution stays in the caller.
        """
        from simWrapper import PolarAction
        self._last_step_index = step
        pending = getattr(self, '_pending_recovery', None)
        if pending is None:
            return PolarAction.null
        if self._recovery_arrived(pending):
            self._finish_checkpoint_recovery(True, pending['reason'])
            return PolarAction.null
        cfg = self.navigation_cfg
        if (pending['ticks'] >= int(cfg.get('rollback_max_steps', 40))
                or int(getattr(self, '_recovery_steps_total', 0)) >= int(cfg.get('rollback_total_step_budget', 60))):
            self._finish_checkpoint_recovery(False, 'Recovery step budget exhausted.')
            return PolarAction.null
        error = self._recovery_error(pending)
        pending['ticks'] += 1
        self._recovery_steps_total = int(getattr(self, '_recovery_steps_total', 0)) + 1
        if (error['horizontal_error_m'] <= float(cfg.get('rollback_position_tolerance', .3))
                and error['height_error_m'] <= float(cfg.get('rollback_height_tolerance_m', .25))):
            angle = self._normalize_angle(pending['target_yaw'] - self._get_agent_yaw())
            return PolarAction(r=0, theta=-angle, type='turn')
        try:
            action = self._get_action_for_next_waypoint()
        except (ValueError, RuntimeError) as exc:
            self._cancel_checkpoint_recovery('Recovery controller failed: ' + type(exc).__name__)
            return PolarAction.null
        if getattr(self, '_pending_recovery', None) is None:
            return PolarAction.null  # The route controller cancelled this transaction.
        if action is None or getattr(action, 'type', None) == 'null':
            self._finish_checkpoint_recovery(False, 'Recovery path ended or became invalid before checkpoint arrival.')
            return PolarAction.null
        return action

    def _observe_recovery_action(self, action):
        """Observe the one production execution record; never duplicate it."""
        pending = getattr(self, '_pending_recovery', None)
        ledger = getattr(self, '_execution_ledger', [])
        if pending is None or len(ledger) <= pending['last_observed_ledger_len']:
            return
        pending['last_observed_ledger_len'] = len(ledger)
        record = ledger[-1]
        if not record.get('recovery'):
            self._finish_checkpoint_recovery(False, 'Recovery execution record lost its transaction marker.')
            return
        if getattr(action, 'type', None) == 'move_forward':
            pending['blocked_moves'] = pending['blocked_moves'] + 1 if float(record.get('motion_distance', 0)) < .03 else 0
        if self._recovery_arrived(pending):
            self._finish_checkpoint_recovery(True, pending['reason'])
        elif pending['blocked_moves'] >= int(self.navigation_cfg.get('rollback_max_blocked_moves', 4)):
            self._finish_checkpoint_recovery(False, 'Recovery movement was repeatedly blocked.')
        elif (pending['ticks'] >= int(self.navigation_cfg.get('rollback_max_steps', 40))
              or int(getattr(self, '_recovery_steps_total', 0)) >= int(self.navigation_cfg.get('rollback_total_step_budget', 60))):
            self._finish_checkpoint_recovery(False, 'Recovery step budget exhausted.')

    def _step_recovery(self, step):
        """Optional select/execute adapter; the normal step UI stays in agent.py."""
        try:
            action = self._recovery_action(step)
            self.execute_action(action)
            self._observe_recovery_action(action)  # Idempotent if execute_action called the hook.
        except BaseException:
            # Runtime budget aborts derive from BaseException. Always release
            # the transaction while preserving the original interruption.
            self._cancel_checkpoint_recovery('Recovery execution was interrupted.')
            raise
        return action

