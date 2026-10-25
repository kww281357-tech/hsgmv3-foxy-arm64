"""Bounded route evidence, endpoint checks and locally scoped branch penalties."""

import copy
import json
import math
import re

import numpy as np

from navigation_contracts import completion_conditions, validate_condition_checks, verification_schema
from navigation_control import check_stair_witnesses, doorway_crossing_required, doorway_witnesses
from agent.endpoint_guard import EndpointGroundingGuard
from spatial_relation import parse_endpoint_relation
from target_identity import detector_matches_endpoint, endpoint_object_roles
from route_progress import route_progress_packet


class RouteEvidenceGuard(EndpointGroundingGuard):
    _DISCRETE_ENDPOINTS = {
        'bed', 'closet', 'wardrobe', 'cabinet', 'cupboard', 'dresser', 'desk',
        'table', 'counter', 'stove', 'oven', 'refrigerator', 'fridge', 'sink',
        'bathtub', 'tub', 'toilet', 'sofa', 'couch', 'chair', 'armchair', 'shelf',
        'shelves', 'bookshelf', 'rug', 'carpet', 'mat', 'fireplace', 'piano', 'tv',
        'television', 'window', 'mirror', 'lamp', 'light', 'chandelier', 'painting',
        'picture', 'sculpture', 'statue', 'vase', 'bar', 'stool', 'barstool',
        'endtable', 'nightstand', 'bench', 'ottoman', 'plant', 'clock', 'door',
        'archway', 'pillar', 'column', 'violin', 'guitar', 'thermostat', 'altar',
        'treadmill', 'banister', 'railing', 'stairs', 'staircase', 'step', 'island',
    }

    _ENDPOINT_ALIASES = {
        'closet': {'closet', 'wardrobe'},
        'wardrobe': {'closet', 'wardrobe'},
        'cabinet': {'cabinet', 'cupboard', 'dresser', 'curio'},
        'cupboard': {'cabinet', 'cupboard', 'dresser'},
        'dresser': {'dresser', 'cabinet', 'chest'},
        'sofa': {'sofa', 'couch'}, 'couch': {'sofa', 'couch'},
        'rug': {'rug', 'carpet', 'mat'}, 'carpet': {'rug', 'carpet', 'mat'},
        'refrigerator': {'refrigerator', 'fridge'}, 'fridge': {'refrigerator', 'fridge'},
        'television': {'television', 'tv'}, 'tv': {'television', 'tv'},
        'stair': {'stair', 'stairs', 'staircase'},
        'staircase': {'stair', 'stairs', 'staircase'},
        'endtable': {'endtable', 'table', 'nightstand'},
        'nightstand': {'nightstand', 'table', 'endtable'},
        'table': {'table', 'desk', 'nightstand', 'endtable', 'counter'},
        'chair': {'chair', 'armchair', 'stool', 'barstool'},
        'armchair': {'armchair', 'chair'},
        'barstool': {'barstool', 'stool', 'chair'},
        'lamp': {'lamp', 'light', 'chandelier'},
        'light': {'light', 'lamp', 'chandelier'},
        'chandelier': {'chandelier', 'lamp', 'light'},
        'banister': {'banister', 'railing', 'handrail'},
        'railing': {'railing', 'banister', 'handrail'},
        'statue': {'statue', 'sculpture'}, 'sculpture': {'statue', 'sculpture'},
    }

    _COMPOUND_ROOTS = {
        'armchair': ('chair',),
        'barstool': ('bar', 'stool', 'chair'),
        'endtable': ('end', 'table'),
        'nightstand': ('table', 'stand'),
        'bookshelf': ('book', 'shelf'),
        'staircase': ('stair', 'stairs'),
        'wardrobe': ('closet',),
        'chandelier': ('lamp', 'light'),
        'fireplace': ('fire',),
        'doorway': ('door',),
        'handrail': ('rail', 'railing', 'banister'),
    }

    @classmethod
    def _normalized_object_words(cls, text):
        cleaned = str(text).replace('_', ' ').lower()
        words = set(re.findall(r'[a-z0-9]+', cleaned))
        normalized = set(words)
        for word in list(words):
            if len(word) > 4 and word.endswith('ies'):
                normalized.add(word[:-3] + 'y')
            elif len(word) > 4 and word.endswith(('ches', 'shes', 'xes', 'zes', 'sses')):
                normalized.add(word[:-2])
            elif len(word) > 3 and word.endswith('s') and not word.endswith('ss'):
                normalized.add(word[:-1])
        for word in list(normalized):
            if word in cls._COMPOUND_ROOTS:
                normalized.update(cls._COMPOUND_ROOTS[word])
        return normalized

    def _endpoint_object_terms(self, target_text):
        words = self._normalized_object_words(target_text)
        targets = words & self._DISCRETE_ENDPOINTS
        return set().union(*(self._ENDPOINT_ALIASES.get(word, {word}) for word in targets)) if targets else set()

    def _route_checks(self, remaining=False):
        obj = self.instruction_obj
        keys = obj.sub_instruction_keys[obj.current_subtask_index:] if remaining else [obj.get_current_subtask_key()]
        checks = []
        for key in keys:
            st = obj.get_subtask_by_key(key)
            if not st:
                continue
            st.setdefault('completion_conditions', completion_conditions(st.get('instruction', ''), key))
            checks.extend(st['completion_conditions'])
        return checks

    def _scene_scope(self):
        return str(getattr(self.sim_wrapper, 'scene_id', '') or '')

    def _episode_start_position(self):
        mapper = getattr(self, 'mapper', None)
        if mapper is not None:
            pcds = getattr(mapper, 'pcds_per_floor', None)
            if isinstance(pcds, dict) and 0 in pcds:
                f0_traj = pcds[0].get('trajectory_position')
                if isinstance(f0_traj, list) and len(f0_traj) > 0:
                    return np.asarray(f0_traj[0], dtype=float)
            traj = getattr(mapper, 'trajectory_position', None)
            if isinstance(traj, list) and len(traj) > 0:
                return np.asarray(traj[0], dtype=float)
        return np.zeros(3, dtype=float)

    def _observed_target_candidates(self, text, limit=10, role='primary'):
        """Detection IDs identify observed instances; they are not goal truth."""
        pose = np.asarray(self.mapper.current_position)
        start_pos = self._episode_start_position()
        full_instr = str(getattr(self, 'instruction', '') or '')
        leaves_start_area = bool(re.search(
            r'\b(exit|leave|out of|down the hall|down the hallway|across|opposite|far end|other end|passing|past|all the way)\b',
            full_instr, re.I
        ))
        rows = []
        for index, ent in enumerate(getattr(self.mapper, 'object_entities', [])):
            name = str(ent.get('class_name', '')).lower()
            if not detector_matches_endpoint(name, text, role):
                continue
            center = np.asarray(ent.get('center', []), dtype=float)
            if center.shape != (3,) or not np.isfinite(center).all():
                continue
            dist_from_start = float(np.linalg.norm((center - start_pos)[:2]))
            delta = center - pose
            center_distance = float(np.linalg.norm(delta[:2]))
            bearing = math.degrees(self._normalize_angle(math.atan2(delta[1], delta[0]) - self._get_agent_yaw()))

            # Compute physical 3D extent and nearest edge distance if point cloud is available
            nearest_distance = center_distance
            extent_info = None
            is_large = False
            pcd = ent.get('pcd')
            if pcd is not None:
                try:
                    pts = pcd.point.positions.cpu().numpy()
                    if len(pts) > 0:
                        min_b = pts.min(axis=0)
                        max_b = pts.max(axis=0)
                        extent = (max_b - min_b)[:2]
                        max_dim = float(np.max(extent))
                        nearest_distance = float(np.min(np.linalg.norm(pts[:, :2] - pose[:2], axis=1)))
                        if max_dim >= 1.5:
                            is_large = True
                            extent_info = [round(float(extent[0]), 2), round(float(extent[1]), 2)]
                except Exception:
                    pass

            # Ignore stale detections far behind the agent (>2.5m and |bearing|>120 deg)
            # that were passed in an earlier room/corridor.
            if nearest_distance > 2.5 and abs(bearing) > 120.0:
                continue

            row = {
                'id': index, 'label': name, 'xy': center[:2].round(2).tolist(),
                'evidence_role': role,
                'horizontal_distance': round(nearest_distance, 2),
                'center_distance': round(center_distance, 2),
                'distance_from_start_m': round(dist_from_start, 2),
                'bearing_degrees': round(bearing, 1),
                'floor': int(getattr(self.mapper, 'floor_level', 0)),
                'confidence': ent.get('confidence'),
                'observation_count': ent.get('observation_count', 0),
                'last_seen_step': ent.get('last_seen_step'),
                'currently_visible': ent.get('last_seen_observation_id') == getattr(self.mapper, 'observation_id', -1),
                'current_regions': [
                    {k: r.get(k) for k in ('view', 'bbox', 'confidence', 'observation_id',
                                          'valid_depth_fraction', 'depth_median_m', 'horizontal_distance_m')}
                    for r in ent.get('observations', [])
                    if r.get('observation_id') == getattr(self.mapper, 'observation_id', -1)],
            }
            if is_large and extent_info:
                row['extent_m'] = extent_info
                row['note'] = f"Large {max(extent_info):.1f}m furniture: nearest edge is {nearest_distance:.2f}m, centroid is {center_distance:.2f}m"
            rows.append(row)

        # Missing detections remain missing. Nearby furniture is not an
        # observed rug instance and must not supply its identity/reachability.
        rows.sort(key=lambda row: row['horizontal_distance'])
        return rows[:limit]

    def _endpoint_clause(self, target_text):
        # The last movement/arrival clause identifies the endpoint; other
        # landmarks in the full segment are supporting route context.
        # Endpoint extraction must not inherit the six-row reporting limit:
        # a long instruction's merged tail may include several transit objects.
        target_checks = completion_conditions(target_text, 'target', max_conditions=128)
        # Precise arrival clause target extraction:
        # If explicit arrival clauses exist (e.g. "stop near the rug"), use ONLY arrival clauses
        # to identify endpoint candidate entities, so transitive/movement objects (e.g. "passing the gray couch")
        # are not misclassified as the destination target.
        arrival_checks = [c for c in target_checks if c.get('requires_arrival', c.get('kind') == 'arrival')]
        endpoint_terms = (self._DISCRETE_ENDPOINTS |
                          {'entrance', 'doorway', 'threshold', 'hallway', 'corridor',
                           'intersection', 'room', 'balcony', 'opening', 'stairs', 'staircase'})
        with_target = [c for c in (arrival_checks or target_checks)
                       if self._normalized_object_words(c['text']) & endpoint_terms]
        if with_target:
            target_text = with_target[-1]['text']
        elif arrival_checks:
            last_arrival = arrival_checks[-1]
            target_text = last_arrival['text']
            if (re.search(r'\b(?:there|here|it|that spot|that place|same spot|same place)\b',
                          target_text, re.I)
                    or re.fullmatch(r'(?:(?:and|then)\s+)?(?:stop|wait|halt|stand)[.!?\s]*',
                                    target_text, re.I)):
                arrival_index = target_checks.index(last_arrival)
                prev_idx = arrival_index - 1
                while prev_idx >= 0 and (arrival_index - prev_idx) <= 2:
                    target_text = target_checks[prev_idx]['text'] + ' ' + target_text
                    if self._normalized_object_words(target_text) & endpoint_terms:
                        break
                    prev_idx -= 1
        else:
            target_text = target_checks[-1]['text'] if target_checks else target_text
        return target_text

    def _prepare_route_verification(self, proposal):
        remaining = False  # Planner output cannot widen a subtask completion check.
        key, st = self.instruction_obj.get_current_subtask()
        checks = self._route_checks(remaining)
        records = (getattr(self, '_execution_ledger', []) if remaining else
                   [r for r in (st or {}).get('record', []) if 'executed_action' in r])
        records = [r for r in records if not r.get('recovery') and not r.get('rejected_by_rollback')]
        # Planner prose is a completion proposal, never authority to change
        # the verification scope or substitute the final segment's target.
        final = remaining or self.instruction_obj.is_last_subtask()
        target_st = self.instruction_obj.get_subtask_by_key(self.instruction_obj.sub_instruction_keys[-1]) if final else st
        target_text = (target_st or {}).get('instruction', '')
        target_text = self._endpoint_clause(target_text)
        if not remaining and st is not None:
            # A verified segment may end on the next segment's threshold. Give
            # the verifier a small, measured handoff window so it can cite that
            # physical crossing without requiring another move past the goal.
            anchor = (st.get('position_anchor_check') or {}).get('status')
            has_own_motion = any(float(r.get('motion_distance', 0.0)) >= 0.03 for r in records)
            needs_translation = any(c.get('requires_translation', c.get('kind') == 'movement') for c in checks)
            if anchor == 'aligned' and self.instruction_obj.current_subtask_index > 0 and (has_own_motion or not needs_translation):
                previous_key = self.instruction_obj.sub_instruction_keys[self.instruction_obj.current_subtask_index - 1]
                previous = self.instruction_obj.get_subtask_by_key(previous_key) or {}
                if previous.get('completed') and previous.get('completion_evidence_reliable'):
                    prior_records = [r for r in previous.get('record', [])
                                     if 'executed_action' in r and not r.get('recovery')
                                     and not r.get('rejected_by_rollback')]
                    records = prior_records[-6:] + records
        # Mid-route progress is separate from subtask completion. Revalidate
        # every saved witness against the live ledger after a rollback.
        saved = {c['id']: c for c in (st or {}).get('route_progress', {}).get('conditions', [])}
        if saved:
            checks, _ = validate_condition_checks(
                checks, [{**c, 'status': 'unknown', 'steps': [], 'evidence': ''} for c in checks],
                records, previous=saved,
                require_instruction_turn=bool(getattr(self, 'navigation_cfg', {}).get('habitat_control_enabled', False)),
                instruction_subtask_id=key)
        keys = getattr(self.instruction_obj, 'sub_instruction_keys', [])
        curr_idx = getattr(self.instruction_obj, 'current_subtask_index', 0)
        next_key = None
        next_instruction = ""
        if not final and curr_idx + 1 < len(keys):
            next_key = keys[curr_idx + 1]
            if hasattr(self.instruction_obj, 'get_subtask_by_key'):
                next_st = self.instruction_obj.get_subtask_by_key(next_key) or {}
                next_instruction = next_st.get('instruction', '')

        verifier_perception = None
        if hasattr(self, '_maybe_refresh_verifier_perception') and callable(getattr(self, '_maybe_refresh_verifier_perception')):
            verifier_perception = self._maybe_refresh_verifier_perception(target_text)

        context = {
            'key': key,
            'curr_instruction': (st or {}).get('instruction', ''),
            'next_key': next_key,
            'next_instruction': next_instruction,
            'remaining': remaining,
            'checks': copy.deepcopy(checks),
            'ledger': list(records),
            'target_text': target_text,
            'endpoint_object_terms': sorted(self._endpoint_object_terms(target_text)),
            'entities': self._observed_target_candidates(target_text) if final else [],
            'reference_entities': (RouteEvidenceGuard._observed_target_candidates(
                self, target_text, role='reference') if final else []),
            'endpoint_object_roles': endpoint_object_roles(target_text),
            'final': final,
            'use_bev': final,
            'observation_id': getattr(self.mapper, 'observation_id', None),
            'verifier_perception': verifier_perception,
        }
        stairs = self._ensure_stair_progress()
        context['stair_progress'] = stairs.packet()
        context['stair_completion_ready'] = stairs.completion_ready()
        self._route_verification_context = context
        return context

    def _route_verification_prompt(self, proposal, clearance, candidate_str=None, depth_and_wp_str=None):
        ctx = self._prepare_route_verification(proposal)
        return self._generate_verification_prompt(ctx, clearance, candidate_str, depth_and_wp_str=depth_and_wp_str)

    def _compact_route_verification_prompt(self, ctx, clearance, candidate_str=None, depth_and_wp_str=None):
        return self._generate_verification_prompt(ctx, clearance, candidate_str, depth_and_wp_str=depth_and_wp_str)

    def _generate_verification_prompt(self, ctx, clearance, candidate_str=None, depth_and_wp_str=None):
        key = ctx.get('key')
        curr_inst = ctx.get('curr_instruction', '')
        next_key = ctx.get('next_key')
        next_inst = ctx.get('next_instruction', '')
        is_final = ctx.get('final', False)
        pos = np.asarray(self.mapper.current_position).round(2).tolist() if hasattr(self, 'mapper') else [0.0, 0.0, 0.0]
        start_p = self._episode_start_position() if hasattr(self, '_episode_start_position') else np.zeros(3, dtype=float)
        curr_p = np.asarray(self.mapper.current_position, dtype=float) if hasattr(self, 'mapper') else np.zeros(3, dtype=float)
        disp_xy = float(np.linalg.norm(curr_p[:2] - start_p[:2]))
        disp_z = float(curr_p[2] - start_p[2])
        traveled_m = float(getattr(self, 'traveled_distance', 0.0) or 0.0)
        pos_str = (
            f"Your current coordinate (z is height): (x: {pos[0]:.2f}, y: {pos[1]:.2f}, z: {pos[2]:.2f}) | "
            f"Episode start: (x: {start_p[0]:.2f}, y: {start_p[1]:.2f}, z: {start_p[2]:.2f}) | "
            f"Net displacement from start: {disp_xy:.2f}m horizontal, {disp_z:+.2f}m vertical (total traveled: {traveled_m:.2f}m)"
        )
        clearance_val = f"{float(clearance):.2f}m" if clearance is not None else "Unknown"
        depth_section = f"\n{depth_and_wp_str}\n" if depth_and_wp_str else ""

        prev_ver_str = ""
        prev_hist = getattr(self, '_verification_history', {}).get(key, [])
        if prev_hist:
            prev_ver_str = (
                '\nPrevious controller-validated verification results (both probes and completion proposals):\n'
                + json.dumps(prev_hist[-2:], ensure_ascii=False)
                + '\nThese are diagnostics, not new route evidence. Address each unknown/contradicted '
                'condition by its ID and diagnostic. Use the actual recorded action steps below; '
                'a planner decision/controller step number is not an execution-ledger step. '
                'Correct invalid citations without claiming an action was newly executed. '
                'Pure endpoint arrival may instead cite the CURRENT observation_id with steps=[]; '
                'movement and turns still need actual executed steps. No action occurred between '
                'same-frame retries: repair citations or image regions without claiming the pose changed. '
                'Endpoint measurements describe only the PREVIOUSLY selected image regions: '
                'mixed_target_depth or insufficient_door_frame_support means the selected region '
                'did not isolate the target/frame; refine the visible target region, not its reported distance. '
                'For split-view doorway support, each region must isolate one jamb, not the whole opening. '
                'A target_surface_too_far result requires approaching the observed target through a legal route; '
                'do not substitute a nearer wall, a different doorway, or the image-center depth. '
                'If the ledger or images cannot establish a condition, leave it unknown and outcome=false; '
                'do not invent evidence or request a physical turn merely to repair a citation.'
            )
            if prev_hist[-1].get('endpoint_confirmation_required') or prev_hist[-1].get('reason') == 'changed_view_confirmation_required':
                prev_ver_str += (
                    '\nIMPORTANT (CHANGED-VIEW CONFIRMATION): The previous verification at this EXACT foot position '
                    'ALREADY grounded the endpoint within stopping distance, and the controller rotated in place '
                    'solely to obtain a second changed-view observation (`changed_view_confirmation_required` is NOT '
                    'a distance failure!). Because the agent has not translated away from the endpoint, if the same '
                    'endpoint is visible in any current view (or has a matching current visual candidate), localize '
                    'its current `target_view` and `target_bbox` (`target_bbox_space="full_view"`) and return '
                    '`outcome=true`, `relation_satisfied=true`, `target_proximity="satisfied"`, and mark the final '
                    'arrival condition `"satisfied"` (with `observation_id` set to the CURRENT observation_id).'
                )

        # Keep the complete bounded episode ledger. A last-N window can hide
        # the doorway crossing or turn that an earlier condition requires.
        recent_records = ctx.get('ledger', [])
        evidence_on = (
            bool(self._evidence_enabled())
            if hasattr(self, '_evidence_enabled') and callable(getattr(self, '_evidence_enabled'))
            else False
        )
        fixed_bev_on = (
            bool(self._fixed_bev_enabled())
            if hasattr(self, '_fixed_bev_enabled') and callable(getattr(self, '_fixed_bev_enabled'))
            else bool((getattr(self, 'navigation_cfg', {}) or {}).get('verifier_fixed_bev', False))
        )
        if fixed_bev_on:
            verifier_map_desc = (
                "Fixed-axis verifier BEV context (+X right, +Y up in world meters; read the on-image legend for START, "
                "prior/current subtask trajectory segments, TURN markers, and CURRENT AGENT POSE arrow; historical/display "
                "context only—cannot supply endpoint grounding or route authority without first-person/ledger proof)."
            )
        else:
            verifier_map_desc = "Explored map memory (agent is the red arrow facing upward; trajectory trail is red)."
        if evidence_on and hasattr(self, '_execution_ledger_text') and callable(getattr(self, '_execution_ledger_text')):
            recent_ledger_str = self._execution_ledger_text(recent_records)
        elif recent_records:
            ledger_lines = []
            for r in recent_records:
                step = r.get('step', '?')
                act = r.get('executed_action', r.get('action_type', 'move'))
                dist = r.get('motion_distance')
                yaw = r.get('yaw_delta')
                parts = [f"step {step}: {act}"]
                if dist is not None:
                    parts.append(f"motion={float(dist):.2f}m")
                if yaw is not None:
                    parts.append(f"yaw={round(math.degrees(float(yaw)), 1)}°")
                if r.get('pose') is not None:
                    parts.append(f"ground_pose={np.asarray(r['pose']).round(2).tolist()}")
                ledger_lines.append(", ".join(parts))
            recent_ledger_str = "\n".join(f"    - {line}" for line in ledger_lines)
        else:
            recent_ledger_str = "    - No executed actions recorded yet."

        cand_str = candidate_str if candidate_str else "None"

        if is_final:
            handoff_rule = (
                "2.  **Final Destination**: This is the final subtask of the navigation route. "
                "There is no subsequent subtask, so return \"handoff_consistent\": null."
            )
            outcome_rule = (
                "3.  **Outcome Gate**: Set \"outcome\": true if and only if \"relation_satisfied\": true "
                "and current visual observations match the instructed destination. Otherwise, set \"outcome\": false."
            )
        else:
            handoff_rule = (
                f"2.  **Handoff Alignment with Next Subtask (`handoff_consistent`)**:\n"
                f"    - Evaluate whether the current position can begin Next Subtask ({next_key}): \"{next_inst}\".\n"
                f"    - Do not require executing the next subtask's turn or movement in advance. A required next turn is not evidence that this segment failed.\n"
                f"    - Return true when the current segment is fulfilled and its endpoint connects to the next segment's start."
            )
            outcome_rule = (
                f"3.  **Outcome Gate**: For this intermediate subtask, return \"outcome\": true ONLY IF "
                f"BOTH \"relation_satisfied\": true AND \"handoff_consistent\": true. If either is false, return \"outcome\": false."
            )

        checks = [{'id': c['id'], 'text': c['text'], 'kind': c['kind']} for c in ctx.get('checks', [])]
        prompt = f"""
You are an intelligent verifier in a simulated indoor environment.
Global Task: {self.instruction}
**Subtask Information**:
- Current Subtask ({key}): "{curr_inst}"
- Next Subtask ({next_key if next_key else "None"}): "{next_inst if next_inst else "None (Final Destination reached)"}"
- Verification Scope: {"Final destination check" if is_final else "Intermediate subtask completion and handoff check"}

**Image Inputs**:
1.  **Past First-Person Keyframes (if provided)**: Earlier front-camera views captured along the route (oldest to newest) showing rooms/landmarks the agent passed.
2.  **Current First-Person Views**: Four separate views tagged [FRONT VIEW (0 deg)], [LEFT VIEW (-90 deg)], [RIGHT VIEW (+90 deg)], and [BACK VIEW (180 deg, behind you)]. Check BACK VIEW and past keyframes to verify what the agent actually passed or crossed!
3.  **Top-Down Map**: {verifier_map_desc}

**Verification Rules**:
1.  **Current Subtask Verification (`relation_satisfied`)**:
    - Check whether the agent has fulfilled ALL sequential clauses, physical motion, path constraints, and stopping relation specified by the Current Subtask ({key}).
    - **TARGET-BOUND DEPTH**: View center/p85 depths describe general surfaces and MUST NOT be called the target's distance. Identify the actual endpoint region in the CURRENT image using target_view and target_bbox, or select its currently visible detector entity. The controller independently measures only that region. A nearby wall/furniture patch cannot supply depth for a different target. Seeing a target in the distance is not arrival.
    - Use the BACK VIEW and PAST KEYFRAMES to verify that intermediate landmarks/doorways (e.g. exiting the starting room and passing a couch or walking down a hallway) were truly passed in the required order. Do NOT mistake furniture/rugs near the episode START for the final destination if the instruction requires leaving the starting room and passing intermediate landmarks first.
    - **SOLID LANDMARKS vs OPEN DOORWAYS/ARCHWAYS**: For solid surface landmarks (window, table, far end of counter/bar, bed, chair, rug), accept arrival when the agent is within ~0.5m-1.2m of the target landmark. For OPENINGS ("stop in the doorway", "stop in the archway"), remember that when the agent stands INSIDE an open doorway or archway threshold, the doorway/archway frame flanks the LEFT/RIGHT views (or was just stepped into) and the room just crossed is in the BACK VIEW, while the FRONT VIEW looks *through* the opening into the next corridor/room (> 1.5m depth, which may show another doorway further ahead) — accept arrival when the agent has crossed the specified hall/room and is standing in that target doorway/archway threshold rather than pushing it into subsequent rooms beyond!
    - If the current subtask is fulfilled, return "relation_satisfied": true. Otherwise, return "relation_satisfied": false.
{handoff_rule}
{outcome_rule}
4.  **Failure Classification & Direct Adjustment Protocol (when outcome=false)**:
    - **Distinguish `"not_reached"` vs `"wrong_path"`**:
      - Set `"failure_type": "not_reached"` and `"scene_consistent": true` when the agent is on the plausible route/hallway/room and simply needs to walk closer (`WP_XX`) or turn (`TURN_*`) to reach the target landmark or threshold.
      - Set `"failure_type": "wrong_path"`, `"scene_consistent": false`, and `"confidence": "high"` only with positive observed evidence contradicting an EXECUTED current or completed route clause. A future clause that has not been attempted is `unknown`, never `contradicted`. Not seeing the final target, facing away from it, needing a turn, or not yet executing later instructions is `not_reached`.
      - For wrong_path, supply `route_contradiction` with kind (`turn_direction`, `straight_path`, `wrong_room`, `wrong_branch`, or `wrong_floor`), exact condition_id, nonempty measured step IDs, expected (quote from that source condition), and observed (positive fact: the opposite executed turn or the room/branch actually entered). Cite the same steps in that condition's check. Do not invent evidence from endpoint absence; use null when there is no contradiction. Scene-consistent routes cannot require rollback.
    - FIRST, provide "correction_reason" (string): Analyze the remaining distance/heading gap (for `not_reached`) or explain why the current room/branch contradicts the route (for `wrong_path`).
    - THEN, provide "correction_action" (object or null): Copy one exact structured navigate or turn action object from the available candidates. Never use bare letters, numbers, coordinates or completion proposals.
    - If outcome=true, return "correction_reason": "" and "correction_action": null.

Do NOT output <think> tags or any prose outside the JSON. Directly output the JSON without markdown code blocks.

**Output Format**:
```json
{{
"observation": "Concise 1-sentence summary of the 4 views and map (<20 words)",
"thought": "Concise 1-sentence check of clause completion and depth evidence (<25 words)",
"relation_satisfied": true/false,  // Whether the current subtask is physically fulfilled
"handoff_consistent": true/false/null,  // Whether current position and heading align with the start of next subtask; null if final subtask
"outcome": true/false,  // Overall verdict: true only if relation_satisfied is true AND (for intermediate subtasks) handoff_consistent is true
"scene_consistent": true/false/null,  // Current views support the instructed room, route and floor; unknown is null
"failure_type": "none",  // "none" if outcome is true; otherwise "not_reached" or "wrong_path"
"confidence": "high",  // Confidence level: "low", "medium", or "high"
"reason": "Concise summary of the verdict and reason",
"contradiction_evidence": "Specific observed route contradiction for wrong_path; empty otherwise",
"route_contradiction": null,  // or {{"kind":"wrong_branch","condition_id":"P1.C2","steps":[12],"expected":"source-clause quote","observed":"positive observed wrong branch"}}
"target_identity": "Describe the instructed endpoint landmark/room visible here",
"target_proximity": "satisfied",  // "satisfied" if outcome is true, else "contradicted" or "unknown"
"target_entity_id": null,  // Integer candidate ID from Observed endpoint candidates if matched, else null
"target_candidate_id": null,  // Fresh current-observation visual candidate_id if matched, else null (detector hint only, not identity truth; 2D candidates may have target_entity_id=null and no metric distance until reliable depth is measured)
"target_view": "front",  // Current view containing the endpoint: front/left/right/back; null if unseen
"target_bbox": [0.1, 0.2, 0.4, 0.8],  // Actual endpoint bounding box [left,top,right,bottom], normalized 0..1 in the original full_view coordinate frame (crop coordinates forbidden); null if unseen. Do not copy this example.
"target_bbox_space": "full_view",  // Must be "full_view" when target_bbox is provided, else null
"reference_region": null,  // {{"view":"right","bbox":[...],"bbox_space":"full_view","entity_id":null,"candidate_id":null,"role":"door_jamb"}} only for the other actual jamb of the SAME doorway, or role="between_object" for B in "between A and B" (uncertain roles use "unknown").
"spatial_landmark_region": null,  // {{"view":"front","bbox":[...],"bbox_space":"full_view","entity_id":null,"candidate_id":null,"role":"spatial_landmark"}} for an independent directional landmark (e.g. statue in "to the left/right of the statue"); never put directional landmarks in reference_region.
"target_evidence": "Visual and depth evidence confirming the stopping relation at the current pose",
"condition_checks": [
    {{"id": "P1.C1", "status": "unknown", "evidence": "State the actual evidence for this clause", "steps": [], "observation_id": null}}
],
"correction_reason": "",  // First explain the distance/heading gap and rationale before micro-adjustment, else empty string
"correction_action": null  // One structured action object from the supplied legal list, or null
}}
```
""".strip()
        prompt += '\nMandatory execution conditions: ' + json.dumps(checks, ensure_ascii=False)
        prompt += '\nMeasured route progress (future conditions remain unknown): ' + json.dumps(route_progress_packet(checks))
        if evidence_on and hasattr(self, '_verification_execution_prompt') and callable(getattr(self, '_verification_execution_prompt')):
            exec_ev_prompt = self._verification_execution_prompt(ctx)
            if exec_ev_prompt:
                prompt += '\n' + str(exec_ev_prompt).lstrip('\n')
        prompt += ('\nMeasured yaw convention: LEFT is negative, RIGHT positive. '
                   'A small alignment turn is not a turn-around; turn-around needs about 180 degrees. '
                   'Each sequential turning instruction needs its own direction-matching executed turn. '
                   'For straight travel cite the start/end executed movement steps of a continuous aligned path, '
                   'including intervening turns; one unrelated forward move is not route completion.')
        prompt += (
            '\nPopulate "condition_checks" with one row per condition ID above. '
            'Cite actual recorded step numbers in "steps". Cite the executed move/turn itself, '
            'not a nearby planner decision number. Arrival may cite an executed step whose '
            'ground_pose matches the CURRENT pose and floor. Only a pure final arrival condition '
            '(no movement or turn requirement) may instead use steps=[] and the CURRENT observation_id '
            'below, even at the initial pose with an empty ledger. All other rows use observation_id=null. '
            'Never invent step 0/1 when no action has executed. Observation references do not prove '
            'motion, crossing, stairs traversal, or a wrong route. Current target identity, localized '
            'metric depth and stopping geometry remain independently required; a whole-image bbox or '
            'one doorway jamb cannot prove the doorway threshold.'
        )
        prompt += '\nCURRENT observation_id (controller-provided, not an execution step): ' + json.dumps(ctx.get('observation_id'))
        door_conditions = [c['id'] for c in ctx.get('checks', []) if doorway_crossing_required(c['text'])]
        if door_conditions:
            prompt += (
                '\nFor door crossing conditions ' + json.dumps(door_conditions) +
                ', add "doorway_checks": [{"condition_id":"exact ID", "steps":[executed crossing step IDs], '
                '"entry_side":"observed pre-crossing room/side", "exit_side":"observed post-crossing room/side", '
                '"visual_evidence":"correct door identity and requested CURRENT side from RGB/BEV"}]. '
                'Use measured before/after poses. Seeing or approaching a door is not crossing it.'
            )
        if is_final:
            prompt += '\nEndpoint clause: ' + ctx.get('target_text', '')
            prompt += '\nDetector category roles (null means no safe category binding): ' + json.dumps(ctx.get('endpoint_object_roles'))
            prompt += (
                '\nLocalize the exact instructed endpoint, not a supporting landmark. '
                'For a doorway include both jambs/frame in target_bbox, not the room seen through it. '
                'When already in the threshold, the jambs may be in DIFFERENT views: localize one '
                'jamb as target_view/target_bbox and the other as reference_region with role="door_jamb". '
                'Only an explicitly identified jamb of the SAME doorway may supply the second frame depth. '
                'A nearby statue or other directional landmark must be localized in spatial_landmark_region '
                'with role="spatial_landmark" (while reference_region remains only the other actual door_jamb or between_object) '
                'and cannot supply doorway geometry. Use original full_view coordinates only (crop coordinates are forbidden). '
                'With missing/unknown/landmark reference roles, target_bbox '
                'must include BOTH doorway borders for independent frame sampling. '
                'For stairs outline the visible stair flight/first risers, not adjacent furniture or a map label. '
                'For between, supply TWO distinct object regions, using role="between_object" for the reference. If no detector candidate matches, '
                'provide your own current-view bbox; this requires a second changed-view confirmation. '
                'Never invent a bbox or copy the schema example when the target is not visible.'
            )
        prompt += (
            f"\n\n**Execution Context**:\n"
            f"BEV tint: blue below and amber above the current floor, in 0.25m bands. It is observed height, not walkability or proof of stair completion.\n"
            f"{pos_str}\n"
            f"Forward Clearance ahead: {clearance_val} (surface ahead, NOT distance to goal).{depth_section}\n"
            f"- Recent executed steps (oldest first):\n"
            f"{recent_ledger_str}{prev_ver_str}\n"
            f"- Available candidate actions for micro-adjustment: {cand_str}\n"
            f"Measured stair phase: {json.dumps(ctx.get('stair_progress', {}))}"
        )
        if is_final:
            prompt += ('\nObserved endpoint candidates (partial detector hints, not identity truth. '
                       'Only currently_visible=true regions may supply current endpoint depth. '
                       'A missing detector candidate requires explicit visual localization, never view-center depth): '
                       + json.dumps(ctx.get('entities', []), ensure_ascii=False))
            prompt += ('\nObserved reference-region candidates (second object for between, '
                       'or the other doorway jamb; never substitute a supporting landmark): '
                       + json.dumps(ctx.get('reference_entities', []), ensure_ascii=False))
        if hasattr(self, '_visual_candidate_prompt') and callable(getattr(self, '_visual_candidate_prompt')):
            prompt += self._visual_candidate_prompt(ctx)
        return prompt


    def _evaluate_route_verification(self, parsed, result):
        ctx = getattr(self, '_route_verification_context', None)
        if not ctx or not isinstance(parsed, dict) or not result.get('available'):
            return result
        if (not result.get('outcome')
                and ctx.get('final')
                and getattr(self, '_pending_endpoint_confirmation', None) is not None
                and parsed.get('scene_consistent') is True
                and parsed.get('failure_type') in (None, 'none', 'not_reached')
                and parsed.get('target_view') in ('front', 'left', 'right', 'back')
                and parsed.get('target_bbox') is not None
                and hasattr(self, 'mapper') and getattr(self.mapper, 'current_position', None) is not None):
            try:
                ep_key = (ctx.get('key'), int(getattr(self.mapper, 'floor_level', 0)))
                prior_obs = (getattr(self, '_endpoint_observations', None) or {}).get(ep_key)
                cur_pose = np.asarray(self.mapper.current_position, dtype=float)
                if (isinstance(prior_obs, dict)
                        and prior_obs.get('pose') is not None
                        and float(np.linalg.norm(cur_pose - np.asarray(prior_obs['pose'], dtype=float))) < 0.05):
                    exp_checks = ctx.get('checks', [])
                    arr_ids = {c['id'] for c in exp_checks if c.get('requires_arrival', c.get('kind') == 'arrival')}
                    raw_ccs = parsed.get('condition_checks')
                    non_arr_ok = (
                        isinstance(raw_ccs, list)
                        and all(
                            isinstance(c, dict) and (c.get('id') in arr_ids or c.get('status') == 'satisfied')
                            for c in raw_ccs
                        )
                    )
                    if non_arr_ok:
                        probe_parsed = copy.deepcopy(parsed)
                        probe_parsed['target_proximity'] = 'satisfied'
                        if not probe_parsed.get('target_bbox_space'):
                            probe_parsed['target_bbox_space'] = 'full_view'
                        probe_res = {'relation_satisfied': True}
                        saved_ep_obs = copy.deepcopy(getattr(self, '_endpoint_observations', {}))
                        saved_final_conf = copy.deepcopy(getattr(self, '_pending_final_stop_confirmation', None))
                        try:
                            probe_reason = self._ground_endpoint_verdict(probe_parsed, ctx, probe_res)
                        finally:
                            self._endpoint_observations = saved_ep_obs
                            self._pending_final_stop_confirmation = saved_final_conf
                        if probe_reason is None and (probe_res.get('endpoint_grounding') or {}).get('valid'):
                            obs_now = ctx.get('observation_id')
                            new_ccs = []
                            for c in raw_ccs:
                                if isinstance(c, dict) and c.get('id') in arr_ids and c.get('status') != 'satisfied':
                                    new_ccs.append({
                                        **c,
                                        'status': 'satisfied',
                                        'steps': [],
                                        'observation_id': obs_now,
                                    })
                                else:
                                    new_ccs.append(c)
                            parsed = {
                                **parsed,
                                'outcome': True,
                                'relation_satisfied': True,
                                'failure_type': 'none',
                                'target_proximity': 'satisfied',
                                'target_bbox_space': probe_parsed['target_bbox_space'],
                                'condition_checks': new_ccs,
                                'correction_action': None,
                                'correction_reason': '',
                            }
                            result.update(
                                outcome=True,
                                relation_satisfied=True,
                                failure_type='none',
                                rollback_required=False,
                                contract_conflict=False,
                                correction_action=None,
                            )
            except Exception:
                pass
        result.update(scope='remaining_route' if ctx['remaining'] else 'current_segment',
                      spatial_clue=str(parsed.get('spatial_clue', '') or '')[:300],
                      rejected_target=str(parsed.get('rejected_target', '') or '')[:100],
                      contract_version='route_evidence_v2', completion_evidence_reliable=False,
                      verifier_perception=ctx.get('verifier_perception'))
        reason = None
        if parsed.get('scene_consistent') is not True and result.get('outcome'):
            reason = 'Route-scene evidence is unknown.'
        elif result.get('contract_conflict') and (result.get('outcome') or not str(result.get('reason') or '').strip()):
            reason = 'Resolve contradictory fields.'
        elif result.get('outcome') and result.get('scene_consistent') is not True:
            reason = 'Confirm route-scene consistency before accepting completion.'
        elif result.get('outcome') and not result.get('relation_satisfied'):
            reason = 'Stopping relation is not satisfied.'
        elif not ctx.get('final') and result.get('outcome') and result.get('handoff_consistent') is not True:
            reason = 'Handoff alignment with next subtask is not satisfied.'
        if result.get('outcome') and reason is None and bool(result.get('negative_landmarks_detected')):
            reason = f"Resolve conflicting objects in scene: {', '.join(str(x) for x in result.get('negative_landmarks_detected', []))}."
        current_st = self.instruction_obj.get_subtask_by_key(ctx['key']) if ctx.get('key') else None
        anchor_check = (current_st or {}).get('position_anchor_check') or {}
        if (result.get('outcome') and reason is None and anchor_check
                and anchor_check.get('status') != 'aligned'):
            reason = 'Current subtask start does not match the previous verified position anchor.'
        if anchor_check:
            result['position_anchor'] = anchor_check
        expected = ctx.get('checks', [])
        observation_id = ctx.get('observation_id')
        if (type(observation_id) is not int or observation_id < 0
                or observation_id != getattr(self.mapper, 'observation_id', None)):
            observation_id = None
        result['current_observation_id'] = observation_id
        result['initial_arrival_observation_eligible'] = bool(
            ctx.get('final') and observation_id is not None and not ctx.get('ledger')
            and len(expected) == 1 and expected[0].get('kind') == 'arrival'
            and expected[0].get('requires_arrival')
            and not expected[0].get('requires_translation') and not expected[0].get('requires_rotation'))
        visual_on = (
            bool(self._visual_enabled())
            if hasattr(self, '_visual_enabled') and callable(getattr(self, '_visual_enabled'))
            else bool((getattr(self, 'navigation_cfg', None) or {}).get('task_visual_enabled', False))
        )
        if ctx.get('final') and visual_on and hasattr(self, '_collect_spatial_relation_observation'):
            self._collect_spatial_relation_observation(parsed, ctx, result)
        arrival_conditions = [c for c in expected if c.get('requires_arrival', c.get('kind') == 'arrival')]
        current_arrival_ids = {arrival_conditions[-1]['id']} if arrival_conditions else set()
        def validate_progress(revoked=()):
            rows, conflict = validate_condition_checks(
                expected, parsed.get('condition_checks'), ctx.get('ledger', []),
                previous={c['id']: c for c in expected}, require_current_arrival=True,
                current_arrival_ids=current_arrival_ids, revoked_condition_ids=revoked,
                current_observation_id=observation_id if ctx.get('final') else None,
                current_pose=np.asarray(self.mapper.current_position) + [0, 0, self.mapper.initial_floor_height],
                require_instruction_turn=bool(getattr(self, 'navigation_cfg', {}).get('habitat_control_enabled', False)),
                instruction_subtask_id=ctx.get('key'))
            return check_stair_witnesses(rows, ctx.get('ledger', [])), conflict

        checked, conflict = validate_progress()
        from route_progress import check_route_contradiction, route_progress_packet, CONTRADICTION_KINDS
        result['route_contradiction'] = None
        claim = parsed.get('route_contradiction')
        structured_request = bool(
            isinstance(claim, dict) and isinstance(claim.get('kind'), str)
            and claim['kind'] in CONTRADICTION_KINDS
            and isinstance(claim.get('condition_id'), str) and claim['condition_id'].strip()
            and isinstance(claim.get('steps'), list) and claim['steps']
            and all(type(step) is int and step >= 0 for step in claim['steps'])
            and all(isinstance(claim.get(key), str) and claim[key].strip() for key in ('expected', 'observed')))
        requested = (parsed.get('failure_type') == 'wrong_path'
                     or bool(result.get('rollback_required')) or structured_request)
        result['route_contradiction_gate'] = {'requested': requested, 'accepted': False,
            'reason': 'not_requested', 'structured_request': structured_request, 'failure_type_inferred': False}
        if requested:
            # A model may challenge a historical milestone, but only a positive
            # structured contradiction can revoke it. Mere absence or changed
            # step citations must not erase a live-ledger-verified prefix.
            cid = claim.get('condition_id') if isinstance(claim, dict) else None
            revoked = {cid} if isinstance(cid, str) else set()
            challenged, challenged_conflict = validate_progress(revoked)
            contradiction, rejection = check_route_contradiction(
                parsed, challenged, ctx.get('ledger', []),
                require_instruction_turn=bool(getattr(self, 'navigation_cfg', {}).get('habitat_control_enabled', False)),
                instruction_subtask_id=ctx.get('key'))
            if contradiction is None:
                reason = rejection
                result['route_contradiction_gate']['reason'] = rejection
                result.update(failure_type='not_reached', rollback_required=False,
                              scene_consistent=(True if parsed.get('scene_consistent') is True else None),
                              negative_landmarks_detected=[])
            elif reason is None:
                control = result
                if structured_request and 'failure_type' not in parsed:
                    # Infer only a missing label, after the existing measured
                    # evidence gate has passed. Reuse the strict verdict
                    # normalizer; prose or a shaped object alone cannot move.
                    from navigation_policy import normalize_verification
                    control = normalize_verification({**copy.deepcopy(parsed), 'failure_type': 'wrong_path'})
                if control.get('rollback_required') and not control.get('contract_conflict'):
                    checked, conflict = challenged, challenged_conflict
                    result.update(failure_type='wrong_path', rollback_required=True, scene_consistent=False,
                                  contradiction_evidence=control.get('contradiction_evidence', ''))
                    result['route_contradiction'] = contradiction
                    result['route_contradiction_gate'].update(accepted=True, reason='verified',
                        failure_type_inferred=structured_request and 'failure_type' not in parsed)
                else:
                    reason = ('Explicit failure_type conflicts with structured route recovery.'
                              if structured_request and 'failure_type' in parsed else
                              'Normalized verdict does not authorize recovery.')
                    result['route_contradiction_gate']['reason'] = reason
            else:
                result['route_contradiction_gate']['reason'] = reason or 'Normalized verdict does not authorize recovery.'
        result['condition_checks'] = checked
        crossings = doorway_witnesses(checked, parsed.get('doorway_checks'), ctx.get('ledger', []))
        result['doorway_checks'] = crossings
        result['stair_progress'] = ctx.get('stair_progress', {})
        result['stair_completion_ready'] = ctx.get('stair_completion_ready')
        progress = route_progress_packet(checked)
        result['route_progress'] = progress
        # Keep only the measured continuous prefix, without promoting a
        # subtask or a terminal relation. Later calls revalidate these records.
        if current_st is not None and not conflict and not result.get('contract_conflict'):
            historical = [dict(c) if c['id'] not in current_arrival_ids
                          else {**c, 'status': 'unknown', 'execution_checked': False}
                          for c in checked]
            historical_progress = route_progress_packet(historical)
            current_st['route_progress'] = {**historical_progress, 'conditions': [
                c for c in historical if c['id'] in historical_progress['completed_prefix']]}
        # Validate the normalized, execution-checked rows, not just the model's
        # top-level verdict or its original spellings (e.g. "failed"). Missing
        # rows, invalid witnesses and unknown endpoint clauses cannot pass.
        result['condition_contract_conflict'] = bool(conflict)
        if result.get('outcome') and reason is None:
            if conflict:
                reason = 'Route condition coverage or execution references are invalid.'
            elif not checked or any(c.get('status') != 'satisfied' for c in checked):
                reason = 'Every route condition requires satisfied execution evidence.'
        if result.get('outcome') and reason is None:
            # StairProgress applies the configured landing policy to the current
            # clause. An absolute/episode-wide height change cannot override it.
            if ctx.get('stair_completion_ready') is False:
                reason = 'Measured stair direction or supported landing is not yet established.'
                # This read checks current support; it must not add a stable
                # sample. Only an executed fresh observation updates StairProgress.
                result['landing_support_current'] = bool(
                    hasattr(self, '_landing_support_observed') and self._landing_support_observed())
            elif ctx.get('final'):
                candidates = {c['id']: c for c in ctx.get('entities', [])}
                raw_target_id = parsed.get('target_entity_id')
                detector_entities = getattr(self.mapper, 'object_entities', [])
                selected_class = (detector_entities[raw_target_id].get('class_name')
                                  if type(raw_target_id) is int and 0 <= raw_target_id < len(detector_entities) else None)
                result['selected_detector_class_name'] = selected_class
                target_id = raw_target_id if (type(raw_target_id) is int and raw_target_id in candidates) else None
                target_id_valid = (raw_target_id is None) or (target_id is not None) or (
                    not getattr(self, 'navigation_cfg', {}).get('require_strict_target_id', True)
                )
                # Detector coverage is partial: a null entity ID is legitimate,
                # but explicit current-view identity/relation evidence is still
                # required. Never fill it from the instruction or generic reason.
                raw_ident = parsed.get('target_identity')
                raw_ident = raw_ident.strip() if isinstance(raw_ident, str) else ''
                raw_prox = parsed.get('target_proximity')
                raw_prox = raw_prox.strip().lower() if isinstance(raw_prox, str) else ''
                raw_evid = parsed.get('target_evidence')
                raw_evid = raw_evid.strip() if isinstance(raw_evid, str) else ''
                result.update(target_entity_id=target_id,
                              target_identity=raw_ident or 'unknown',
                              target_proximity=raw_prox or 'unknown',
                              target_evidence=raw_evid[:600])
                target_ident_valid = raw_ident.lower() not in ('', 'unknown', 'none', 'unidentified', 'contradicted', 'null', 'false', 'no', 'failed')
                target_prox_valid = raw_prox in ('satisfied', 'reached', 'close', 'true', 'yes')
                if not target_id_valid or not target_ident_valid or not target_prox_valid or not raw_evid:
                    reason = 'Current endpoint identity and stopping relation are not established.'
                elif (getattr(self, 'navigation_cfg', {}).get('require_endpoint_grounding', True)
                      or getattr(self, 'navigation_cfg', {}).get('task_visual_enabled', False)
                      or any(c.get('observation_checked') for c in checked)):
                    reason = self._ground_endpoint_verdict(parsed, ctx, result)
                    ep_pkt = result.get('endpoint_grounding') if isinstance(result.get('endpoint_grounding'), dict) else {}
                    result['target_candidate_id'] = ep_pkt.get('candidate_id') if ep_pkt.get('valid') else None
                    result['target_bbox_space'] = ep_pkt.get('bbox_space') if ep_pkt.get('valid') else None
                    spatial_reason = (
                        self._verify_spatial_relation(parsed, ctx, result)
                        if hasattr(self, '_verify_spatial_relation')
                        else None
                    )
                    if reason is None:
                        reason = spatial_reason
        if result.get('outcome') and reason is None:
            strict_doorway = getattr(self, 'navigation_cfg', {}).get('require_strict_doorway_checks', True)
            if strict_doorway and any(not c.get('execution_witnessed') for c in crossings):
                reason = 'Doorway crossing requires measured before/after poses and observed entry/exit sides.'
        if result.get('outcome') and reason is None and ctx.get('final'):
            # Endpoint grounding: a named object must be detected near the stop point.
            # Disabled by default (max_grounding_vetoes=0): in the p0 test the detector missed a
            # large table the agent was standing next to and the veto turned a success into a timeout.
            weak_terms = {'door', 'stairs', 'stair', 'staircase', 'step'}
            terms = set(ctx.get('endpoint_object_terms') or []) - weak_terms
            if terms:
                near = [c for c in ctx.get('entities', []) if float(c.get('horizontal_distance', 99)) <= 3.5]
                vetoes = getattr(self, '_grounding_vetoes', None)
                if vetoes is None:
                    vetoes = self._grounding_vetoes = {}
                used = vetoes.get(ctx.get('key'), 0)
                if not near and used < int(self.navigation_cfg.get('max_grounding_vetoes', 0)):
                    vetoes[ctx.get('key')] = used + 1
                    reason = ('No detected ' + '/'.join(sorted(terms)[:3]) + ' within 3.5 m of the current position; '
                              'look for or move to the named endpoint object before stopping.')
        if reason and result.get('outcome'):
            import logging
            logging.info(f"[VerifierGate] Overturned affirmative verdict for '{ctx.get('key')}': {reason}")
        if reason:
            result.update(outcome=False, failure_type='not_reached', rollback_required=False,
                          relation_satisfied=False, confidence='low', reason=reason)
        result['evidence_state'] = ('satisfied' if result.get('outcome') else
                                    'contradicted' if result.get('rollback_required') else
                                    'unknown' if reason else 'contradicted')
        result['completion_evidence_reliable'] = bool(
            result.get('outcome') and result.get('relation_satisfied') and not result.get('contract_conflict')
            and not conflict and checked and all(c.get('status') == 'satisfied' for c in checked)
            and (ctx.get('final') or result.get('handoff_consistent') is True)
        )
        if not ctx['remaining']:
            _, st = self.instruction_obj.get_current_subtask()
            if st is not None and not result.get('contract_conflict'):
                st['completion_evidence_reliable'] = result['completion_evidence_reliable']
                if result['completion_evidence_reliable']:
                    st['completion_conditions'] = checked
        return result

    def _is_near_endpoint_or_threshold(self, at_route_boundary=False):
        """Check if agent is currently in the braking zone near the endpoint, threshold, or junction."""
        if not hasattr(self, 'instruction_obj') or not self.instruction_obj:
            return False
        cfg = getattr(self, 'navigation_cfg', {})
        if (cfg.get('habitat_control_enabled', False) and hasattr(self, '_active_turn_clause')
                and hasattr(self, '_executed_instruction_turn')):
            active_turn = self._active_turn_clause()
            if active_turn is not None and self._executed_instruction_turn(active_turn) is None:
                return False
        if callable(getattr(self, '_current_turn_departure', None)) and self._current_turn_departure() is not None:
            return False
        final_key = self.instruction_obj.sub_instruction_keys[-1] if self.instruction_obj.sub_instruction_keys else None
        st = self.instruction_obj.get_subtask_by_key(final_key) if final_key else None
        st_text = self._endpoint_clause(str((st or {}).get('instruction', ''))).lower()
        threshold_words = {'entrance', 'door', 'archway', 'threshold', 'intersect', 'intersection',
                           'halfway', 'between', 'opening', 'exit', 'stairs', 'staircase', 'landing'}
        is_threshold = any(w in st_text for w in threshold_words)

        pose = np.asarray(self.mapper.current_position)
        target_pos = getattr(self, 'target_world_position', None)
        target_dist = float(np.linalg.norm(np.asarray(target_pos)[:2] - pose[:2])) if target_pos is not None else 999.0

        candidates = self._observed_target_candidates(st_text, limit=3)
        min_candidate_dist = min(
            (float(c.get('horizontal_distance', c.get('center_distance', 999.0))) for c in candidates),
            default=999.0,
        )
        rel_spec = parse_endpoint_relation(st_text)
        spatial_landmark_near = True
        if rel_spec.get('required') and rel_spec.get('landmark_text'):
            lm_cands = self._observed_target_candidates(rel_spec['landmark_text'], limit=3, role='primary')
            min_lm_dist = min(
                (float(c.get('horizontal_distance', c.get('center_distance', 999.0))) for c in lm_cands),
                default=999.0,
            )
            spatial_landmark_near = min_lm_dist <= 3.0

        path_len = 0.0
        curr_path = getattr(self, 'current_path', [])
        if curr_path:
            p = [pose[:2]] + [np.asarray(pt)[:2] for pt in curr_path]
            path_len = sum(float(np.linalg.norm(p[i] - p[i-1])) for i in range(1, len(p)))

        if spatial_landmark_near and min_candidate_dist <= float(self.navigation_cfg.get('arrival_probe_near_distance_m', 2.0)):
            return True
        # A planned waypoint is not the annotated endpoint. Only treat it as
        # a threshold candidate when the final instruction actually names a
        # boundary, and only at the end of that planned move.
        return bool(is_threshold and target_dist <= 1.2
                    and ((curr_path and len(curr_path) == 1 and path_len <= 1.2) or at_route_boundary))

    @staticmethod
    def _same_terminal_instance(first, second):
        """An explicit identity switch cannot corroborate the first STOP claim.

        Missing detector IDs remain unknown, handled by the visual contract;
        this check does not fabricate IDs or equate unknown with a mismatch.
        """
        a, b = first.get('target_entity_id'), second.get('target_entity_id')
        return a is None or b is None or str(a) == str(b)

    def _terminal_probe_due(self, step, stalled=False, at_route_boundary=False):
        """Schedule a bounded current-segment check from observed endpoint cues.

        The caller must claim only when it actually dispatches a check. This
        predicate never changes scope, commits progress or reads evaluator goals.
        """
        cfg = getattr(self, 'navigation_cfg', {})
        obj = getattr(self, 'instruction_obj', None)
        if not cfg.get('arrival_check_enabled', False) or obj is None or not obj.is_last_subtask():
            return False
        if getattr(self, 'rolling_back', False) or getattr(self, '_pending_recovery', None):
            return False
        if (cfg.get('habitat_control_enabled', False) and hasattr(self, '_active_turn_clause')
                and hasattr(self, '_executed_instruction_turn')):
            active_turn = self._active_turn_clause()
            if active_turn is not None and self._executed_instruction_turn(active_turn) is None:
                return False
        if (cfg.get('habitat_control_enabled', False) and hasattr(self, '_habitat_terminal_candidate')
                and self._habitat_terminal_candidate()):
            # A measured foot-level structural arrival must interrupt a live
            # route immediately; the old periodic interval can walk past it.
            # Identity/route evidence and bounded fresh confirmation follow.
            return True
        key, subtask = obj.get_current_subtask()
        needs_motion = any(c.get('requires_translation') for c in completion_conditions((subtask or {}).get('instruction', ''), key))
        if needs_motion and not any(r.get('action_type') == 'move_forward' and float(r.get('motion_distance', 0)) >= .02
                                    and not r.get('recovery') and not r.get('rejected_by_rollback')
                                    for r in (subtask or {}).get('record', [])):
            return False
        if not self._is_near_endpoint_or_threshold(at_route_boundary=at_route_boundary):
            return False
        key = obj.get_current_subtask_key()
        state = getattr(self, '_terminal_probe_checks', {}).get(key, {})
        if int(state.get('calls', 0)) >= max(0, int(cfg.get('arrival_probe_max_calls', 4))):
            return False
        if not state:
            return True
        elapsed = int(step) - int(state['step'])
        if elapsed < max(1, int(cfg.get('arrival_probe_min_interval_steps', 2))):
            return False
        pose = np.asarray(self.mapper.current_position, dtype=float)
        moved = float(np.linalg.norm(pose - np.asarray(state['pose'], dtype=float)))
        rotation = np.asarray(self.mapper.current_rotation, dtype=float)
        previous_rotation = np.asarray(state['rotation'], dtype=float)
        angle = 0.0
        if rotation.shape == (3, 3) and previous_rotation.shape == (3, 3):
            cosine = float((np.trace(rotation @ previous_rotation.T) - 1.0) / 2.0)
            angle = math.acos(float(np.clip(cosine, -1.0, 1.0)))
        changed_view = (
            moved >= float(cfg.get('arrival_probe_min_distance_m', .35))
            or angle >= math.radians(float(cfg.get('arrival_probe_min_turn_degrees', 30.0)))
        )
        fresh_observation = getattr(self, '_observation_revision', 0) != state.get('revision')
        observation_due = elapsed >= max(1, int(cfg.get('arrival_probe_observation_interval_steps', 8)))
        # Stalling can justify another observation after the longer interval,
        # but cannot turn repeated identical observations into new evidence.
        return bool(changed_view or (fresh_observation and observation_due
                                    and (stalled or at_route_boundary)))

    def _claim_terminal_probe(self, step):
        """Consume one proactive check immediately before its verifier call."""
        key = self.instruction_obj.get_current_subtask_key()
        checks = getattr(self, '_terminal_probe_checks', None)
        if checks is None:
            checks = self._terminal_probe_checks = {}
        state = checks.get(key, {})
        checks[key] = {
            'calls': int(state.get('calls', 0)) + 1,
            'step': int(step),
            'pose': np.asarray(self.mapper.current_position, dtype=float).copy(),
            'rotation': np.asarray(self.mapper.current_rotation, dtype=float).copy(),
            'revision': getattr(self, '_observation_revision', 0),
        }

    def _commit_remaining_route(self, verification):
        """The controller alone commits evidenced progress and the single final STOP."""
        if not verification.get('completion_evidence_reliable') or verification.get('scope') != 'remaining_route':
            raise ValueError('Remaining-route completion requires an accepted execution contract.')
        obj = self.instruction_obj
        keys = obj.sub_instruction_keys[obj.current_subtask_index:]
        checks = {c['id']: c for c in verification.get('condition_checks', [])}
        for key in keys:
            st = obj.get_subtask_by_key(key)
            if 'completion_conditions' in st and checks:
                st['completion_conditions'] = [checks[c['id']] for c in st['completion_conditions'] if c['id'] in checks]
            st['completion_evidence_reliable'] = True
            st['progress_reconciled_from_execution'] = True
            obj.update_verifier({**verification, 'subtask_id': key})
            prefix = f'{key}.'
            cited_steps = [int(step) for condition_id, row in checks.items()
                           if condition_id.startswith(prefix)
                           for step in row.get('steps', []) if type(step) is int]
            latest_step = max(cited_steps, default=None)
            anchor_pose = None
            if latest_step is not None:
                record = next((r for r in reversed(getattr(self, '_execution_ledger', []))
                               if int(r.get('step', -1)) == latest_step
                               and r.get('subtask_id') == key), None)
                if record is not None and record.get('pose') is not None:
                    anchor_pose = record['pose']
            obj.mark_current_subtask_completed(
                self.mapper.current_position - [0, 0, 1.2], self.mapper.current_rotation,
                anchor_pose=anchor_pose)
            if hasattr(self, '_commit_completed_subtask_memory'):
                self._commit_completed_subtask_memory(key)
        self.navigation_trace.append({'role': 'controller', 'event': 'route_progress_reconciled',
                                      'segments': keys, 'evidence': verification.get('evidence', '')})

    def _remember_failed_branch(self, checkpoint, reason, strong=False):
        if not self.navigation_cfg.get('branch_constraints_enabled', True) or checkpoint.get('target') is None:
            return
        origin = np.asarray(checkpoint['start_position'], dtype=float)
        vector = np.asarray(checkpoint['target'], dtype=float)[:2] - origin[:2]
        if np.linalg.norm(vector) < 0.15:
            return
        angle = math.atan2(vector[1], vector[0])
        rows = getattr(self, '_failed_branches', [])
        decision = getattr(self, '_navigation_decision_id', 0)
        for row in rows:
            if (row['subtask_id'] == checkpoint['subtask_id'] and row['floor'] == checkpoint['floor_level']
                    and np.linalg.norm(np.asarray(row['origin'])[:2] - origin[:2]) < 0.6
                    and abs(self._normalize_angle(row['angle'] - angle)) < math.radians(20)):
                row.update(hits=row['hits']+1, expires=decision+12, strong=bool(row['strong'] or strong), reason=reason)
                break
        else:
            rows.append({'subtask_id': checkpoint['subtask_id'], 'floor': checkpoint['floor_level'],
                         'origin': origin.tolist(), 'angle': angle, 'hits': 1, 'strong': strong,
                         'expires': decision+12, 'reason': str(reason)[:240]})
        self._failed_branches = rows[-16:]

    def _constrain_branch_actions(self, actions):
        key = self.instruction_obj.get_current_subtask_key()
        floor = getattr(self.mapper, 'floor_level', 0)
        pose = np.asarray(self.mapper.current_position) - [0, 0, 1.2]
        decision = getattr(self, '_navigation_decision_id', 0)
        rows = [r for r in getattr(self, '_failed_branches', []) if r['expires'] >= decision]
        self._failed_branches = rows
        penalties = {}
        excluded = set()
        for action_id, action in actions.items():
            if action.get('type') != 'waypoint':
                continue
            target = np.asarray(action['target_point_world'])
            for row in rows:
                if row['subtask_id'] != key or row['floor'] != floor:
                    continue
                origin = np.asarray(row['origin'])
                if np.linalg.norm(pose[:2] - origin[:2]) > 1.2:
                    continue
                vector = target[:2] - origin[:2]
                if np.linalg.norm(vector) < 0.3:
                    continue
                angle = math.atan2(vector[1], vector[0])
                if abs(self._normalize_angle(angle - row['angle'])) <= math.radians(25):
                    penalties[str(action_id)] = max(penalties.get(str(action_id), 0), row['hits'] + int(row['strong']))
                    if row['strong']:
                        excluded.add(str(action_id))
        # One evidenced wrong-path event is sufficient to avoid re-entering
        # that branch. Keep turns when all visible waypoints are excluded so
        # the planner can observe alternatives. Uncertain events stay advisory.
        self._branch_action_penalties = penalties
        return {k: a for k, a in actions.items() if str(k) not in excluded}

