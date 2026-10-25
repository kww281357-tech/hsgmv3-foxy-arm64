import math
from PIL import Image, ImageDraw, ImageFont

BG_COLOR = (18, 24, 38)
PANEL_BG = (26, 34, 52)
PANEL_BORDER = (51, 65, 85)
TEXT_COLOR = (226, 232, 240)
MUTED_TEXT = (148, 163, 184)
OBS_COLOR = (100, 116, 139)
CURRENT_POSE_COLOR = (34, 197, 94)
CURRENT_ARROW_COLOR = (74, 222, 128)
TURN_BEFORE_COLOR = (251, 191, 36)
TURN_AFTER_COLOR = (34, 211, 238)
STYLE_COLORS = {
    "executed": (59, 130, 246),
    "recovery": (245, 158, 11),
    "rejected": (239, 68, 68),
    "blocked": (192, 132, 252),
    "turn": (34, 211, 238),
}
MAX_ABS_COORD = 1e7
MIN_MAX_SIDE = 240


def _is_bool_like(val):
    if isinstance(val, bool):
        return True
    t_name = type(val).__name__.lower()
    return t_name in ("bool", "bool_", "boolean")


def _valid_int(val, min_val=0):
    if _is_bool_like(val) or not isinstance(val, int):
        return False
    return val >= min_val


def _parse_num(val, limit=MAX_ABS_COORD):
    if val is None or _is_bool_like(val):
        return None
    if isinstance(val, int):
        if abs(val) > limit:
            return None
        return float(val)
    if not isinstance(val, float):
        t_mod = getattr(type(val), "__module__", "") or ""
        if not t_mod.startswith("numpy"):
            return None
    try:
        f = float(val)
    except (OverflowError, ValueError, TypeError):
        return None
    if not math.isfinite(f) or abs(f) > limit:
        return None
    return f


def _parse_pose(pose):
    if isinstance(pose, (str, bytes, dict)) or not hasattr(pose, "__len__"):
        return None
    if len(pose) < 2:
        return None
    x = _parse_num(pose[0])
    y = _parse_num(pose[1])
    if x is None or y is None:
        return None
    z = 0.0
    has_z = False
    if len(pose) >= 3 and pose[2] is not None:
        z = _parse_num(pose[2])
        if z is None:
            return None
        has_z = True
    return (x, y, z, has_z)


def _unit_px(yaw):
    return [round(math.cos(yaw), 6), round(-math.sin(yaw), 6)]


def _draw_arrow(draw, x0, y0, dx, dy, length, color, width=2):
    x1 = int(round(x0 + dx * length))
    y1 = int(round(y0 + dy * length))
    draw.line([(x0, y0), (x1, y1)], fill=color, width=width)
    head = max(4, int(length * 0.32))
    px, py = -dy, dx
    lx = int(round(x1 - dx * head + px * head * 0.55))
    ly = int(round(y1 - dy * head + py * head * 0.55))
    rx = int(round(x1 - dx * head - px * head * 0.55))
    ry = int(round(y1 - dy * head - py * head * 0.55))
    draw.polygon([(x1, y1), (lx, ly), (rx, ry)], fill=color)
    return [x1, y1]


def _draw_move_arrow(draw, x0, y0, x1, y1, color):
    draw.line([(x0, y0), (x1, y1)], fill=color, width=3)
    dx, dy = x1 - x0, y1 - y0
    seg_len = math.hypot(dx, dy)
    if seg_len < 1e-6:
        return False
    ux, uy = dx / seg_len, dy / seg_len
    head = min(max(4, int(round(seg_len * 0.28))), 9)
    px, py = -uy, ux
    lx = int(round(x1 - ux * head + px * head * 0.55))
    ly = int(round(y1 - uy * head + py * head * 0.55))
    rx = int(round(x1 - ux * head - px * head * 0.55))
    ry = int(round(y1 - uy * head - py * head * 0.55))
    draw.polygon([(x1, y1), (lx, ly), (rx, ry)], fill=color)
    return True


def _clamp_xy(x, y, bounds):
    xmin, xmax, ymin, ymax = bounds
    cx = min(max(x, xmin), xmax)
    cy = min(max(y, ymin), ymax)
    return cx, cy, (cx != x or cy != y)


def _select_anchor(cur_p, retained, sampled_pts, radius):
    if cur_p is not None:
        return cur_p[0], cur_p[1]
    if retained:
        pts = []
        for _, p0, p1, _, _, _ in retained:
            pts.append((p0[0], p0[1]))
            pts.append((p1[0], p1[1]))
        best_cluster = [pts[0]]
        best_count = -1
        for cx, cy in pts:
            cluster = [
                (px, py)
                for px, py in pts
                if abs(px - cx) <= radius and abs(py - cy) <= radius
            ]
            if len(cluster) > best_count:
                best_count = len(cluster)
                best_cluster = cluster
        xs = sorted(p[0] for p in best_cluster)
        ys = sorted(p[1] for p in best_cluster)
        mid = (len(xs) - 1) // 2
        return xs[mid], ys[mid]
    if sampled_pts:
        return sampled_pts[0][0], sampled_pts[0][1]
    return 0.0, 0.0


def _place_map_label(pt_px, text, panel_bounds, forbidden_pts, used_boxes):
    mx0, my0, mx1, my1 = panel_bounds
    tw = max(12, 6 * len(text))
    th = 10
    offsets = [
        (5, -12),
        (5, 4),
        (-tw - 5, -12),
        (-tw - 5, 4),
        (6, -20),
        (-tw - 6, -20),
        (6, 12),
        (-tw - 6, 12),
    ]
    best_box = None
    for dx, dy in offsets:
        lx = min(max(pt_px[0] + dx, mx0 + 4), max(mx0 + 4, mx1 - tw - 4))
        ly = min(max(pt_px[1] + dy, my0 + 13), max(my0 + 13, my1 - th - 13))
        box = [lx, ly, lx + tw, ly + th]
        if best_box is None:
            best_box = box
        hit_pt = any(
            (box[0] - 2 <= fx <= box[2] + 2 and box[1] - 2 <= fy <= box[3] + 2)
            for fx, fy in forbidden_pts
        )
        if hit_pt:
            continue
        hit_box = any(
            not (box[2] < ub[0] or box[0] > ub[2] or box[3] < ub[1] or box[1] > ub[3])
            for ub in used_boxes
        )
        if not hit_box:
            return box
    return best_box


def render_verification_bev(
    ledger,
    current_pose,
    current_yaw,
    observed_points=None,
    max_side=672,
    *,
    max_records=128,
    max_points=2000,
    max_turn_labels=8,
    max_step_labels=12,
    outlier_radius=15.0,
):
    if not _valid_int(max_side, MIN_MAX_SIDE):
        raise ValueError(f"max_side must be an int >= {MIN_MAX_SIDE}")
    if not _valid_int(max_records, 1):
        raise ValueError("max_records must be an int >= 1")
    if not _valid_int(max_points, 0):
        raise ValueError("max_points must be an int >= 0")
    if not _valid_int(max_turn_labels, 1):
        raise ValueError("max_turn_labels must be an int >= 1")
    if not _valid_int(max_step_labels, 1):
        raise ValueError("max_step_labels must be an int >= 1")
    radius = _parse_num(outlier_radius, limit=1e6)
    if radius is None or radius <= 0:
        raise ValueError("outlier_radius must be a positive finite number")

    raw_rows = (
        list(ledger)
        if ledger is not None and not isinstance(ledger, (str, bytes, dict))
        else []
    )
    valid_rows = []
    invalid_rows = 0
    for row in raw_rows:
        if not isinstance(row, dict) or not _valid_int(row.get("step"), 0):
            invalid_rows += 1
            continue
        p0 = _parse_pose(row.get("pose_before"))
        p1 = _parse_pose(row.get("pose"))
        if p0 is None or p1 is None:
            invalid_rows += 1
            continue
        yb = _parse_num(row.get("yaw_before"), limit=1e5)
        ya = _parse_num(row.get("yaw_after"), limit=1e5)
        yd = _parse_num(row.get("yaw_delta"), limit=1e5)
        if yd is None and yb is not None and ya is not None:
            yd = math.atan2(math.sin(ya - yb), math.cos(ya - yb))
        valid_rows.append((row, p0, p1, yb, ya, yd))

    omitted_rows = valid_rows[:-max_records] if len(valid_rows) > max_records else []
    retained = valid_rows[-max_records:] if len(valid_rows) > max_records else valid_rows
    omitted_steps = [r[0]["step"] for r in omitted_rows]

    raw_pts = (
        list(observed_points)
        if observed_points is not None
        and not isinstance(observed_points, (str, bytes, dict))
        else []
    )
    valid_pts = []
    invalid_pts = 0
    for pt in raw_pts:
        parsed = _parse_pose(pt)
        if parsed is None:
            invalid_pts += 1
        else:
            valid_pts.append((parsed[0], parsed[1]))
    if max_points == 0:
        sampled_pts = []
    elif len(valid_pts) > max_points:
        n_pts = len(valid_pts)
        sampled_pts = [
            valid_pts[int(i * n_pts // max_points)] for i in range(max_points)
        ]
    else:
        sampled_pts = list(valid_pts)
    omitted_pts = len(valid_pts) - len(sampled_pts)

    cur_p = _parse_pose(current_pose)
    cur_y = _parse_num(current_yaw, limit=1e5)
    anchor_x, anchor_y = _select_anchor(cur_p, retained, sampled_pts, radius)

    win_bounds = (
        anchor_x - radius,
        anchor_x + radius,
        anchor_y - radius,
        anchor_y + radius,
    )
    inlier_xs, inlier_ys = ([cur_p[0]], [cur_p[1]]) if cur_p is not None else ([], [])
    for _, p0, p1, _, _, _ in retained:
        for px, py in ((p0[0], p0[1]), (p1[0], p1[1])):
            if abs(px - anchor_x) <= radius and abs(py - anchor_y) <= radius:
                inlier_xs.append(px)
                inlier_ys.append(py)
    for px, py in sampled_pts:
        if abs(px - anchor_x) <= radius and abs(py - anchor_y) <= radius:
            inlier_xs.append(px)
            inlier_ys.append(py)
    if not inlier_xs:
        inlier_xs, inlier_ys = [anchor_x], [anchor_y]

    cx_mid = 0.5 * (min(inlier_xs) + max(inlier_xs))
    cy_mid = 0.5 * (min(inlier_ys) + max(inlier_ys))
    span = (
        max(max(inlier_xs) - min(inlier_xs), max(inlier_ys) - min(inlier_ys), 1.6)
        * 1.25
    )
    half = 0.5 * span
    view_bounds = (cx_mid - half, cx_mid + half, cy_mid - half, cy_mid + half)

    img = Image.new("RGB", (max_side, max_side), BG_COLOR)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    pad = max(6, int(max_side * 0.025))
    gap = max(2, int(max_side * 0.01))
    hdr_h = max(15, int(max_side * 0.048))
    leg_h = max(36, int(max_side * 0.11))
    turn_h = max(36, int(max_side * 0.14))
    z_h = max(32, int(max_side * 0.13))
    avail_map_h = max_side - hdr_h - leg_h - turn_h - z_h - 4 * gap - pad
    map_side = max(80, min(max_side - 2 * pad, avail_map_h))

    mx0 = (max_side - map_side) // 2
    my0 = hdr_h
    mx1, my1 = mx0 + map_side, my0 + map_side
    draw.rectangle([mx0, my0, mx1, my1], fill=PANEL_BG, outline=PANEL_BORDER)
    draw.text((pad, 2), "BEV VERIFICATION (+x right, +y up)", fill=TEXT_COLOR, font=font)
    draw.text((mx1 - 18, my0 + map_side // 2 - 5), "+x", fill=MUTED_TEXT, font=font)
    draw.text((mx1 - 18, my0 + 2), "+y", fill=MUTED_TEXT, font=font)

    inner_pad = max(14, int(map_side * 0.08))
    plot_side = max(24, map_side - 2 * inner_pad)
    px_left, py_top = mx0 + inner_pad, my0 + inner_pad
    scale = plot_side / span

    def to_px(wx, wy):
        u = (wx - view_bounds[0]) / span
        v = (view_bounds[3] - wy) / span
        return [int(round(px_left + u * plot_side)), int(round(py_top + v * plot_side))]

    clipped_pts = 0
    for ox, oy in sampled_pts:
        cx, cy, was_c = _clamp_xy(ox, oy, win_bounds)
        cx, cy, was_v = _clamp_xy(cx, cy, view_bounds)
        if was_c or was_v:
            clipped_pts += 1
        ppx, ppy = to_px(cx, cy)
        draw.rectangle([ppx - 1, ppy - 1, ppx + 1, ppy + 1], fill=OBS_COLOR)

    rendered_rows, turn_events, clipped_steps, gap_steps = [], [], [], []
    move_row_indices = []
    forbidden_pts = []
    prev_step = None
    prev_s_px1 = None

    for idx, (row, p0, p1, yb, ya, yd) in enumerate(retained):
        step = row["step"]
        if prev_step is not None and step != prev_step + 1:
            gap_steps.append([prev_step, step])
        prev_step = step
        x0, y0, c0 = _clamp_xy(p0[0], p0[1], win_bounds)
        x1, y1, c1 = _clamp_xy(p1[0], p1[1], win_bounds)
        x0, y0, v0 = _clamp_xy(x0, y0, view_bounds)
        x1, y1, v1 = _clamp_xy(x1, y1, view_bounds)
        row_clipped = bool(c0 or c1 or v0 or v1)
        if row_clipped:
            clipped_steps.append(step)

        rej = bool(row.get("rejected_by_rollback") or row.get("rejected"))
        rec = bool(row.get("recovery"))
        blk = bool(row.get("blocked"))
        dist_xy = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        has_turn = (
            yb is not None
            and ya is not None
            and (row.get("action_type") == "turn" or abs(yd or 0.0) > 1e-4)
        )
        if rej:
            style = "rejected"
        elif rec:
            style = "recovery"
        elif blk:
            style = "blocked"
        elif has_turn and dist_xy < 1e-3:
            style = "turn"
        else:
            style = "executed"
        color = STYLE_COLORS[style]
        s_px0, s_px1 = to_px(x0, y0), to_px(x1, y1)

        has_move_arrow = False
        if dist_xy < 1e-3 or s_px0 == s_px1:
            draw.ellipse(
                [s_px0[0] - 2, s_px0[1] - 2, s_px0[0] + 2, s_px0[1] + 2], fill=color
            )
        else:
            has_move_arrow = _draw_move_arrow(
                draw, s_px0[0], s_px0[1], s_px1[0], s_px1[1], color
            )

        mid_px = ((s_px0[0] + s_px1[0]) // 2, (s_px0[1] + s_px1[1]) // 2)
        forbidden_pts.extend([tuple(s_px0), tuple(s_px1), mid_px])
        if prev_s_px1 is not None:
            forbidden_pts.append(
                ((prev_s_px1[0] + s_px0[0]) // 2, (prev_s_px1[1] + s_px0[1]) // 2)
            )
        prev_s_px1 = s_px1

        if dist_xy >= 1e-3 and not has_turn:
            move_row_indices.append(idx)

        is_instr = bool(row.get("instruction_turn")) and has_turn and abs(yd or 0.0) > 1e-4
        t_label = "instruction_turn" if is_instr else ("turn" if has_turn else None)
        t_dir = (
            "right"
            if (yd or 0.0) > 1e-4
            else ("left" if (yd or 0.0) < -1e-4 else "none")
        )
        rec_src_raw = row.get("recovery_source")
        rec_src = (
            rec_src_raw.strip()
            if (rec and isinstance(rec_src_raw, str) and rec_src_raw.strip())
            else ("unknown" if rec else "none")
        )
        db_px = _unit_px(yb) if yb is not None else None
        da_px = _unit_px(ya) if ya is not None else None

        if has_turn:
            on_map = len(turn_events) < max_turn_labels
            turn_events.append(
                {
                    "step": step,
                    "label": t_label,
                    "turn_direction": t_dir,
                    "yaw_before": round(yb, 6),
                    "yaw_after": round(ya, 6),
                    "yaw_delta": round(yd or 0.0, 6),
                    "dir_before_px": db_px,
                    "dir_after_px": da_px,
                    "label_drawn_on_map": on_map,
                    "strip_box_px": None,
                }
            )

        rendered_rows.append(
            {
                "step": step,
                "action_type": str(row.get("action_type") or "unknown"),
                "style": style,
                "color_rgb": list(color),
                "segment_xy": [
                    [round(p0[0], 5), round(p0[1], 5)],
                    [round(p1[0], 5), round(p1[1], 5)],
                ],
                "segment_px": [s_px0, s_px1],
                "move_arrow": has_move_arrow,
                "z_before": round(p0[2], 5) if p0[3] else None,
                "z_after": round(p1[2], 5) if p1[3] else None,
                "yaw_before": round(yb, 6) if yb is not None else None,
                "yaw_after": round(ya, 6) if ya is not None else None,
                "yaw_delta": round(yd, 6) if yd is not None else None,
                "dir_before_px": db_px,
                "dir_after_px": da_px,
                "turn_direction": t_dir,
                "turn_label": t_label,
                "recovery": rec,
                "recovery_source": rec_src,
                "rejected_by_rollback": rej,
                "blocked": blk,
                "clipped": row_clipped,
                "independent_segment": True,
            }
        )

    if cur_p is not None:
        cx_c, cy_c, _ = _clamp_xy(cur_p[0], cur_p[1], view_bounds)
        c_px_forbid = to_px(cx_c, cy_c)
        forbidden_pts.append(tuple(c_px_forbid))
        forbidden_pts.append((c_px_forbid[0], c_px_forbid[1] - 12))
        forbidden_pts.append((c_px_forbid[0] + 12, c_px_forbid[1]))

    selected_move_idx = set(move_row_indices[-max_step_labels:])
    omitted_step_labels = max(0, len(move_row_indices) - len(selected_move_idx))
    turn_on_map_steps = {t["step"] for t in turn_events if t["label_drawn_on_map"]}
    used_label_boxes = []
    label_boxes_px = {}
    step_labeled_steps = []
    turn_labeled_steps = []
    map_panel_bounds = (mx0, my0, mx1, my1)

    for idx, r in enumerate(rendered_rows):
        step = r["step"]
        is_turn_lbl = step in turn_on_map_steps
        is_move_lbl = idx in selected_move_idx
        if not (is_turn_lbl or is_move_lbl):
            continue
        lbl_text = f"s{step}"
        lbl_color = TURN_AFTER_COLOR if is_turn_lbl else TEXT_COLOR
        box = _place_map_label(
            r["segment_px"][1], lbl_text, map_panel_bounds, forbidden_pts, used_label_boxes
        )
        draw.text((box[0], box[1]), lbl_text, fill=lbl_color, font=font)
        used_label_boxes.append(box)
        label_boxes_px[str(step)] = box
        if is_turn_lbl:
            turn_labeled_steps.append(step)
        if is_move_lbl:
            step_labeled_steps.append(step)

    current_arrow = None
    if cur_p is not None:
        cx, cy, _ = _clamp_xy(cur_p[0], cur_p[1], view_bounds)
        c_px = to_px(cx, cy)
        draw.ellipse(
            [c_px[0] - 4, c_px[1] - 4, c_px[0] + 4, c_px[1] + 4],
            fill=CURRENT_POSE_COLOR,
        )
        if cur_y is not None:
            dx, dy = math.cos(cur_y), -math.sin(cur_y)
            arr_len = max(14, int(plot_side * 0.12))
            tip_px = _draw_arrow(
                draw, c_px[0], c_px[1], dx, dy, arr_len, CURRENT_ARROW_COLOR, width=3
            )
            current_arrow = {
                "pose_xy": [round(cur_p[0], 5), round(cur_p[1], 5)],
                "pose_z": round(cur_p[2], 5) if cur_p[3] else None,
                "yaw_rad": round(cur_y, 6),
                "direction_px": [round(dx, 6), round(dy, 6)],
                "base_px": c_px,
                "tip_px": tip_px,
            }
    else:
        draw.text(
            (mx0 + 4, my1 - 12),
            "CURRENT POSE: UNKNOWN",
            fill=STYLE_COLORS["rejected"],
            font=font,
        )

    clipping_active = bool(clipped_steps or clipped_pts)
    clip_banner_box = None
    clip_banner_text = None
    if clipping_active:
        step_tag = ",".join(f"s{s}" for s in clipped_steps[:3])
        if len(clipped_steps) > 3:
            step_tag += ".."
        prefix = f"CLIP {step_tag}:" if step_tag else "CLIP:"
        clip_banner_text = f"{prefix}{len(clipped_steps)}r,{clipped_pts}p"
        cb_x0, cb_y0 = mx0 + 4, my0 + 2
        cb_x1 = min(mx1 - 22, cb_x0 + 6 * len(clip_banner_text))
        cb_y1 = cb_y0 + 10
        draw.text((cb_x0, cb_y0), clip_banner_text, fill=TURN_BEFORE_COLOR, font=font)
        clip_banner_box = [cb_x0, cb_y0, cb_x1, cb_y1]

    ly0 = my1 + gap
    ly1 = ly0 + leg_h
    lx0, lx1 = pad, max_side - pad
    draw.rectangle([lx0, ly0, lx1, ly1], fill=PANEL_BG, outline=PANEL_BORDER)
    leg_hdr = (
        "LEGEND | OBS samples: neutral (not navigable)"
        if (lx1 - lx0) >= 280
        else "LEGEND | OBS: neutral, not navigable"
    )
    draw.text((lx0 + 4, ly0 + 1), leg_hdr, fill=MUTED_TEXT, font=font)

    legend_entries = [
        ("executed", "executed", STYLE_COLORS["executed"]),
        ("recovery", "recovery", STYLE_COLORS["recovery"]),
        ("rejected", "rejected", STYLE_COLORS["rejected"]),
        ("blocked", "blocked", STYLE_COLORS["blocked"]),
        ("current", "current", CURRENT_POSE_COLOR),
        ("observed_samples", "obs(neutral)", OBS_COLOR),
    ]
    leg_col_w = max(40, (lx1 - lx0 - 8) // 3)
    legend_meta = {}
    for idx, (k, short_lbl, col) in enumerate(legend_entries):
        r_idx = idx // 3
        c_idx = idx % 3
        ix0 = lx0 + 4 + c_idx * leg_col_w
        iy0 = ly0 + 13 + r_idx * 11
        sw_box = [ix0, iy0 + 1, ix0 + 7, iy0 + 8]
        draw.rectangle(sw_box, fill=col)
        draw.text((ix0 + 10, iy0 - 1), short_lbl, fill=TEXT_COLOR, font=font)
        legend_meta[k] = {
            "color_rgb": list(col),
            "swatch_box_px": sw_box,
            "label": short_lbl,
        }

    ty0 = ly1 + gap
    ty1 = ty0 + turn_h
    tx0, tx1 = pad, max_side - pad
    draw.rectangle([tx0, ty0, tx1, ty1], fill=PANEL_BG, outline=PANEL_BORDER)
    draw.text((tx0 + 4, ty0 + 2), "TURN", fill=MUTED_TEXT, font=font)
    tb_sw = [tx0 + 34, ty0 + 3, tx0 + 41, ty0 + 10]
    draw.rectangle(tb_sw, fill=TURN_BEFORE_COLOR)
    draw.text((tx0 + 44, ty0 + 2), "before", fill=TEXT_COLOR, font=font)
    ta_sw = [tx0 + 86, ty0 + 3, tx0 + 93, ty0 + 10]
    draw.rectangle(ta_sw, fill=TURN_AFTER_COLOR)
    draw.text((tx0 + 96, ty0 + 2), "after", fill=TEXT_COLOR, font=font)

    legend_meta["turn"] = {
        "color_rgb": list(STYLE_COLORS["turn"]),
        "swatch_box_px": ta_sw,
        "label": "turn",
    }
    legend_meta["turn_before"] = {
        "color_rgb": list(TURN_BEFORE_COLOR),
        "swatch_box_px": tb_sw,
        "label": "before",
    }
    legend_meta["turn_after"] = {
        "color_rgb": list(TURN_AFTER_COLOR),
        "swatch_box_px": ta_sw,
        "label": "after",
    }

    avail_turn_w = max(24, (tx1 - 4) - (tx0 + 4))
    max_fit_turns = max(1, avail_turn_w // 22)
    n_show_turns = min(len(turn_events), max_turn_labels, max_fit_turns)
    shown_turns = turn_events[-n_show_turns:] if n_show_turns > 0 else []
    if shown_turns:
        slot_w = avail_turn_w / max(1, len(shown_turns))
        for idx, tev in enumerate(shown_turns):
            bx0 = int(round(tx0 + 4 + idx * slot_w))
            bx1 = int(round(tx0 + 4 + (idx + 1) * slot_w)) - 2
            bx1 = min(tx1 - 4, max(bx0 + 8, bx1))
            by0 = ty0 + 13
            by1 = max(by0 + 10, ty1 - 3)
            draw.rectangle([bx0, by0, bx1, by1], outline=PANEL_BORDER)
            tcx, tcy = (bx0 + bx1) // 2, (by0 + by1) // 2 + 2
            alen = max(5, min(bx1 - bx0, by1 - by0) // 3)
            b_tip = _draw_arrow(
                draw,
                tcx,
                tcy,
                tev["dir_before_px"][0],
                tev["dir_before_px"][1],
                alen,
                TURN_BEFORE_COLOR,
                width=2,
            )
            a_tip = _draw_arrow(
                draw,
                tcx,
                tcy,
                tev["dir_after_px"][0],
                tev["dir_after_px"][1],
                alen,
                TURN_AFTER_COLOR,
                width=2,
            )
            draw.text((bx0 + 2, by0 + 1), f"s{tev['step']}", fill=TEXT_COLOR, font=font)
            tev["strip_box_px"] = [bx0, by0, bx1, by1]
            tev["before_tip_px"] = b_tip
            tev["after_tip_px"] = a_tip

    zy0 = ty1 + gap
    zy1 = max_side - pad
    zx0, zx1 = pad, max_side - pad
    draw.rectangle([zx0, zy0, zx1, zy1], fill=PANEL_BG, outline=PANEL_BORDER)
    z_vals = [
        r["z_before"] for r in rendered_rows if r["z_before"] is not None
    ] + [r["z_after"] for r in rendered_rows if r["z_after"] is not None]
    if cur_p is not None and cur_p[3]:
        z_vals.append(round(cur_p[2], 5))
    z_min = min(z_vals) if z_vals else 0.0
    z_max = max(z_vals) if z_vals else 0.0
    z_span = max(z_max - z_min, 0.5)
    if rendered_rows:
        z_hdr = (
            f"HEIGHT z (m): [{z_min:.2f}, {z_max:.2f}] "
            f"s{rendered_rows[0]['step']}..s{rendered_rows[-1]['step']}"
        )
    else:
        z_hdr = f"HEIGHT z (m): [{z_min:.2f}, {z_max:.2f}]"
    draw.text((zx0 + 4, zy0 + 1), z_hdr, fill=MUTED_TEXT, font=font)

    z_segs = []
    z_labeled_steps = []
    n_ret = len(rendered_rows)
    z_left, z_right = zx0 + 6, zx1 - 6
    z_top, z_bot = zy0 + 13, max(zy0 + 18, zy1 - 4)
    if n_ret > 0 and z_right > z_left:
        avail_z_w = z_right - z_left
        col_w = avail_z_w / max(1, n_ret)
        plot_h = max(4, z_bot - z_top)
        for i, r in enumerate(rendered_rows):
            if r["z_before"] is None or r["z_after"] is None:
                continue
            x0_f = z_left + i * col_w
            x1_f = z_left + (i + 0.75) * col_w
            sx0 = min(max(int(round(x0_f)), z_left), z_right)
            sx1 = min(max(int(round(x1_f)), sx0), z_right)
            if col_w >= 2.0 and sx1 == sx0 and sx0 < z_right:
                sx1 = sx0 + 1
            sy0 = int(round(z_bot - ((r["z_before"] - z_min) / z_span) * plot_h))
            sy1 = int(round(z_bot - ((r["z_after"] - z_min) / z_span) * plot_h))
            sy0 = min(max(sy0, z_top), z_bot)
            sy1 = min(max(sy1, z_top), z_bot)
            draw.line([(sx0, sy0), (sx1, sy1)], fill=tuple(r["color_rgb"]), width=2)
            z_segs.append(
                {
                    "step": r["step"],
                    "z_before": r["z_before"],
                    "z_after": r["z_after"],
                    "dz": round(r["z_after"] - r["z_before"], 5),
                    "px0": [sx0, sy0],
                    "px1": [sx1, sy1],
                }
            )
        if z_segs:
            cand_indices = [0, len(z_segs) // 2, len(z_segs) - 1]
            last_lbl_x = -999
            for c_idx in cand_indices:
                seg = z_segs[c_idx]
                if seg["step"] in z_labeled_steps:
                    continue
                st_txt = f"s{seg['step']}"
                tx = min(max(seg["px0"][0], z_left), max(z_left, z_right - 6 * len(st_txt)))
                if tx - last_lbl_x >= 6 * len(st_txt) + 4:
                    draw.text((tx, max(zy0 + 11, z_bot - 9)), st_txt, fill=TEXT_COLOR, font=font)
                    z_labeled_steps.append(seg["step"])
                    last_lbl_x = tx

    omitted_turn_labels = sum(1 for t in turn_events if not t["label_drawn_on_map"])
    omitted_map_labels = omitted_turn_labels + omitted_step_labels
    move_arrow_steps = [r["step"] for r in rendered_rows if r["move_arrow"]]

    meta = {
        "image_size": [max_side, max_side],
        "axes": {
            "x_positive": "right",
            "y_positive": "up",
            "pixel_y_positive": "down",
            "yaw_convention": "atan2(forward_y, forward_x)",
            "arrow_pixel_vector": "[cos(yaw), -sin(yaw)]",
            "compass_north": None,
            "labels": ["+x", "+y", "z (m)"],
        },
        "transform": {
            "x_min": round(view_bounds[0], 5),
            "x_max": round(view_bounds[1], 5),
            "y_min": round(view_bounds[2], 5),
            "y_max": round(view_bounds[3], 5),
            "scale_px_per_m": round(scale, 5),
            "map_box_px": [px_left, py_top, px_left + plot_side, py_top + plot_side],
            "panel_box_px": [mx0, my0, mx1, my1],
            "origin_px": to_px(0.0, 0.0),
        },
        "current_pose_valid": cur_p is not None,
        "current_yaw_valid": cur_y is not None,
        "current_arrow": current_arrow,
        "counts": {
            "total_ledger_rows": len(raw_rows),
            "retained_rows": len(rendered_rows),
            "omitted_history_rows": len(omitted_rows),
            "omitted_steps": omitted_steps,
            "invalid_rows": invalid_rows,
            "missing_step_gaps": gap_steps,
            "total_observed_points": len(raw_pts),
            "rendered_observed_points": len(sampled_pts),
            "omitted_observed_points": omitted_pts,
            "invalid_observed_points": invalid_pts,
            "clipped_steps": clipped_steps,
            "clipped_points_count": clipped_pts,
            "turn_label_limit": max_turn_labels,
            "omitted_turn_labels": omitted_turn_labels,
            "step_label_limit": max_step_labels,
            "omitted_step_labels": omitted_step_labels,
            "omitted_map_labels": omitted_map_labels,
            "move_arrow_count": len(move_arrow_steps),
        },
        "map_labels": {
            "move_arrow_steps": move_arrow_steps,
            "step_labeled_steps": step_labeled_steps,
            "turn_labeled_steps": turn_labeled_steps,
            "label_boxes_px": label_boxes_px,
            "omitted_step_labels": omitted_step_labels,
            "omitted_turn_labels": omitted_turn_labels,
            "omitted_map_labels": omitted_map_labels,
        },
        "coordinate_clipping": {
            "outlier_radius_m": round(radius, 4),
            "clipped_steps": clipped_steps,
            "clipped_row_count": len(clipped_steps),
            "clipped_point_count": clipped_pts,
            "active": clipping_active,
            "banner_box_px": clip_banner_box,
            "banner_text": clip_banner_text,
        },
        "rendered_rows": rendered_rows,
        "row_styles": {str(r["step"]): r["style"] for r in rendered_rows},
        "turn_events": turn_events,
        "z_profile": {
            "unit": "m",
            "min_z": round(z_min, 5),
            "max_z": round(z_max, 5),
            "current_z": round(cur_p[2], 5) if (cur_p is not None and cur_p[3]) else None,
            "panel_box_px": [zx0, zy0, zx1, zy1],
            "plot_box_px": [z_left, z_top, z_right, z_bot],
            "labeled_steps": z_labeled_steps,
            "segments": z_segs,
            "gap_steps": gap_steps,
        },
        "observed_samples": {
            "label": "OBSERVED samples (geometry-neutral, not navigable)",
            "semantics": "observed_only_not_free_space_or_gt",
            "rendered_count": len(sampled_pts),
            "omitted_count": omitted_pts,
            "invalid_count": invalid_pts,
            "clipped_count": clipped_pts,
        },
        "legend_panel_box_px": [lx0, ly0, lx1, ly1],
        "legend": legend_meta,
    }
    return img, meta

