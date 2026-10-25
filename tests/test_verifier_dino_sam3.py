import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from segmentation.verifier_dino_sam3 import (
    build_verifier_dino_prompt_terms,
    refresh_verifier_candidates_with_dino_sam3,
    verifier_dino_sam3_enabled,
)


def test_verifier_dino_sam3_enabled_flag_and_env():
    assert verifier_dino_sam3_enabled({}, {}) is False
    assert verifier_dino_sam3_enabled({"verifier_dino_sam3_enabled": True}, {}) is True
    assert verifier_dino_sam3_enabled({"verifier_dino_sam3_enabled": False}, {"HSGM_VERIFIER_DINO_SAM3": "1"}) is True
    assert verifier_dino_sam3_enabled({"verifier_dino_sam3_enabled": True}, {"HSGM_VERIFIER_DINO_SAM3": "0"}) is False


def test_build_verifier_dino_prompt_terms_prioritizes_task_terms():
    terms = build_verifier_dino_prompt_terms(
        instruction_text="Walk straight and stop in the doorway to the left of the white statue of two people.",
        endpoint_text="stop in the doorway to the left of the white statue of two people",
        turn_text="Turn right at the eye chart",
        max_terms=24,
    )
    assert any("statue" in t for t in terms[:8])
    assert any("doorway" in t or "door" in t for t in terms[:8])
    assert any("eye chart" in t or "chart" in t for t in terms[:10])
    assert len(terms) <= 24

    focused = build_verifier_dino_prompt_terms(
        instruction_text="Walk straight and stop in the doorway to the left of the white statue of two people.",
        endpoint_text="stop in the doorway to the left of the white statue of two people",
        turn_text="",
        max_terms=8,
        task_only=True,
    )
    assert "white statue" in focused or "statue" in focused
    assert "doorway" in focused
    assert "toilet" not in focused
    assert "towel" not in focused


def test_refresh_verifier_candidates_with_dino_sam3_caches_per_observation(monkeypatch):
    import segmentation.verifier_dino_sam3 as vds

    calls = []

    def fake_segment_view(rgb, terms, **kwargs):
        calls.append((rgb.shape, tuple(terms[:3])))
        mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        mask[10:30, 10:30] = 255
        return [
            {
                "class_id": 1,
                "class_name": "white statue",
                "canonical_label": "statue",
                "mask": mask,
                "conf": 0.78,
                "detector_source": "dino_sam3",
            }
        ]

    monkeypatch.setattr(vds, "dino_sam3_segment_view", fake_segment_view)

    class DummyMapper:
        def __init__(self):
            self.observation_id = 5
            self.current_visual_candidates = {
                "5:front:0": {"candidate_id": "5:front:0", "observation_id": 5, "detector_source": "yoloe"}
            }

        def refresh_current_views_with_segmenter(self, seg_fn):
            rgb = np.zeros((64, 64, 3), dtype=np.uint8)
            dets = seg_fn(rgb, "front")
            self.current_visual_candidates = {
                "5:front:0": {
                    "candidate_id": "5:front:0",
                    "observation_id": 5,
                    "detector_class_name": dets[0]["class_name"],
                    "canonical_label": dets[0]["canonical_label"],
                    "confidence": dets[0]["conf"],
                }
            }
            return dets

    class DummyAgent:
        navigation_cfg = {"verifier_dino_sam3_enabled": True}
        instruction = "stop in the doorway to the left of the white statue"
        mapper = DummyMapper()

    res1 = refresh_verifier_candidates_with_dino_sam3(DummyAgent, "doorway to the left of the white statue")
    assert res1["enabled"] is True
    assert res1["refreshed"] is True
    assert res1["cached"] is False
    assert len(calls) == 1
    assert DummyAgent.mapper.current_visual_candidates["5:front:0"]["detector_source"] == "dino_sam3"

    # Second call on the same observation_id must use cache without re-running segmenter
    res2 = refresh_verifier_candidates_with_dino_sam3(DummyAgent, "doorway to the left of the white statue")
    assert res2["cached"] is True
    assert len(calls) == 1


def test_refresh_verifier_candidates_with_dino_sam3_supplements_only_missing_terms(monkeypatch):
    import segmentation.verifier_dino_sam3 as vds

    queried_terms = []

    def fake_segment_view(rgb, terms, **kwargs):
        queried_terms.append(tuple(terms))
        mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        mask[10:30, 10:30] = 255
        return [
            {
                "class_id": 1,
                "class_name": terms[0],
                "canonical_label": terms[0],
                "mask": mask,
                "conf": 0.75,
                "detector_source": "dino_sam3",
            }
        ]

    monkeypatch.setattr(vds, "dino_sam3_segment_view", fake_segment_view)

    class DummyMapper:
        def __init__(self, current_visual_candidates):
            self.observation_id = 7
            self.current_visual_candidates = current_visual_candidates

        def refresh_current_views_with_segmenter(self, seg_fn):
            rgb = np.zeros((64, 64, 3), dtype=np.uint8)
            return seg_fn(rgb, "front")

    class DummyAgent:
        navigation_cfg = {"verifier_dino_sam3_enabled": True}
        instruction = "stop in the doorway to the left of the white statue"
        mapper = DummyMapper(
            {
                "7:front:0": {
                    "candidate_id": "7:front:0",
                    "observation_id": 7,
                    "detector_class_name": "door",
                    "canonical_label": "door",
                    "detector_source": "yoloe",
                }
            }
        )

    # YOLOE already detected doorway ('door'), so DINO+SAM3 should only query missing statue terms
    res_partial = refresh_verifier_candidates_with_dino_sam3(
        DummyAgent, "doorway to the left of the white statue"
    )
    assert res_partial["refreshed"] is True
    assert queried_terms == [("white statue", "statue")]

    # If YOLOE already detected both 'door' and 'white statue', DINO+SAM3 should skip completely
    DummyAgent.mapper = DummyMapper(
        {
            "7:front:0": {
                "candidate_id": "7:front:0",
                "observation_id": 7,
                "detector_class_name": "door",
                "canonical_label": "door",
                "detector_source": "yoloe",
            },
            "7:front:1": {
                "candidate_id": "7:front:1",
                "observation_id": 7,
                "detector_class_name": "white statue",
                "canonical_label": "white statue",
                "detector_source": "yoloe",
            },
        }
    )
    DummyAgent._verifier_dino_sam3_last_obs = None
    queried_terms.clear()
    res_full = refresh_verifier_candidates_with_dino_sam3(
        DummyAgent, "doorway to the left of the white statue"
    )
    assert res_full["refreshed"] is False
    assert res_full["reason"] == "all_task_terms_detected_by_yoloe"
    assert queried_terms == []

