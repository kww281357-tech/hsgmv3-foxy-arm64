import ast
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from segmentation.object_list import categories
from semantic_evidence import canonical_label, rgb_to_bgr
from target_identity import detector_matches_endpoint, endpoint_object_roles, object_category
from task_visual import bounded_instruction_terms, is_equivalent_label


class _FakeTensor:
    def __init__(self, array):
        self._array = np.asarray(array)

    def cpu(self):
        return self

    def numpy(self):
        return self._array

    def item(self):
        return self._array.item()


class _FakeMasks:
    def __init__(self, array):
        self.data = _FakeTensor(array)


class _FakeBoxes:
    def __init__(self, cls_ids, confs):
        self.cls = [_FakeTensor(c) for c in cls_ids]
        self.conf = [_FakeTensor(c) for c in confs]


class _FakeResult:
    def __init__(self, masks, cls_ids, confs, names):
        self.masks = _FakeMasks(masks) if masks is not None else None
        self.boxes = _FakeBoxes(cls_ids, confs)
        self.names = dict(names)


class _FakeCV2:
    INTER_NEAREST = 0

    @staticmethod
    def resize(mask, size, interpolation=0):
        width, height = size
        ys = np.linspace(0, mask.shape[0] - 1, height).astype(int)
        xs = np.linspace(0, mask.shape[1] - 1, width).astype(int)
        return mask[np.ix_(ys, xs)]


class _FakeYOLOE:
    def __init__(self, path=""):
        self.path = path
        self.text_pe_calls = []
        self.set_classes_calls = []
        self.predict_calls = []
        self.active_classes = []
        self.next_class_id = 0

    def get_text_pe(self, prompts):
        prompts = list(prompts)
        self.text_pe_calls.append(prompts)
        return {"prompts": tuple(prompts), "call_index": len(self.text_pe_calls)}

    def set_classes(self, class_names, text_pe):
        self.active_classes = list(class_names)
        self.set_classes_calls.append((list(class_names), text_pe))

    def predict(self, image, **kwargs):
        self.predict_calls.append((np.asarray(image).copy(), dict(kwargs)))
        mask = np.zeros((1, 4, 4), dtype=np.uint8)
        mask[0, 1:3, 1:3] = 1
        names = {i: name for i, name in enumerate(self.active_classes)}
        return [_FakeResult(mask, [self.next_class_id], [0.91], names)]


def _load_segmentation_harness():
    source_path = SRC / "segmentation" / "instance_segmentation.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    mod_ast = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(mod_ast)
    ns = {
        "__file__": str(source_path),
        "os": __import__("os"),
        "np": np,
        "cv2": _FakeCV2,
        "YOLOE": _FakeYOLOE,
        "categories": categories,
        "canonical_label": canonical_label,
        "rgb_to_bgr": rgb_to_bgr,
        "bounded_instruction_terms": bounded_instruction_terms,
        "is_equivalent_label": is_equivalent_label,
    }
    exec(compile(mod_ast, str(source_path), "exec"), ns)
    return ns


def test_canonical_label_and_target_identity_eye_chart_aliases_and_exclusions():
    for alias in ("eye chart", "eye_chart", "eye test chart", "Snellen chart", "snellen_charts"):
        assert canonical_label(alias) == "eye_chart"
        assert object_category(alias) == "eyechart"
        assert detector_matches_endpoint("eye_chart", f"Walk forward and stop at the {alias}.")
        assert detector_matches_endpoint(alias, "Wait in front of the eye chart.")

    assert endpoint_object_roles("Stand between the Snellen chart and the sink") == {
        "primary": "eyechart",
        "reference": "sink",
    }
    assert object_category("wall beside the sofa") is None
    assert object_category("the wall near the eye chart") is None
    assert object_category("door handle") is None
    assert object_category("cabinet door") is None
    assert object_category("trophy case") is None
    assert object_category("custom sign") is None
    assert object_category("table lamp") == "lamp"
    assert object_category("table lamp") != "table"
    assert not detector_matches_endpoint("sofa", "stop at the wall beside the sofa")
    assert not detector_matches_endpoint("eye_chart", "stop at the wall near the eye chart")
    assert not detector_matches_endpoint("door", "stop at the cabinet door")
    assert not detector_matches_endpoint("door", "stop at the door handle")
    assert not detector_matches_endpoint("table", "stop at the table lamp")
    assert detector_matches_endpoint("lamp", "stop at the table lamp")
    assert detector_matches_endpoint("door", "Stop at the door to the left of the trophy case")
    assert not detector_matches_endpoint("trophy case", "Stop at the door to the left of the trophy case")


def test_bounded_instruction_terms_Compounds_and_route_exclusions():
    assert bounded_instruction_terms("Stop at the door to the left of the trophy case") == [
        "door",
        "trophy case",
    ]
    assert bounded_instruction_terms("Go through the door next to the fire extinguisher") == [
        "door",
        "fire extinguisher",
    ]
    assert bounded_instruction_terms("Walk past the custom sign and stop at the door") == [
        "custom sign",
        "door",
    ]
    assert bounded_instruction_terms(
        "Turn left at the cabinet door, pass the door handle, and stop at the Snellen chart near the table lamp."
    ) == ["eye chart", "cabinet door", "door handle", "table lamp"]

    terms = bounded_instruction_terms(
        "Walk out of the bedroom, go down the hallway, turn left at the cabinet door, "
        "turn right at the Snellen chart, and stop by the table lamp beside the bed."
    )
    assert terms == ["eye chart", "cabinet door", "table lamp", "bed"]
    assert bounded_instruction_terms("door handle") == ["door handle"]
    assert bounded_instruction_terms("cabinet door") == ["cabinet door"]
    assert bounded_instruction_terms("table lamp") == ["table lamp"]
    assert bounded_instruction_terms("stop at the wall beside the sofa") == ["sofa"]
    assert bounded_instruction_terms("stop at the wall near the eye chart") == ["eye chart"]
    assert bounded_instruction_terms("Walk down the hallway and go through the room") == []
    assert bounded_instruction_terms(
        "Turn at the eye test chart, wait by the Snellen chart, and stop at the eye_chart."
    ) == ["eye chart"]

    many = ", ".join(f"item{i} chart" for i in range(25))
    assert len(bounded_instruction_terms(many, max_extra_terms=5)) == 5
    assert len(bounded_instruction_terms(many)) == 16
    assert len(bounded_instruction_terms(many, max_extra_terms=100)) == 16


def test_configure_task_vocabulary_cache_reset_and_four_view_predict():
    ns = _load_segmentation_harness()
    model = ns["model"]
    configure = ns["configure_task_vocabulary"]
    segment = ns["instance_segmentation"]
    base_labels = [cat["name"] for cat in categories]

    assert len(base_labels) == 240
    assert model.active_classes == base_labels
    assert len(model.text_pe_calls) == 1
    assert len(model.set_classes_calls) == 1

    meta1 = configure(
        "Turn left at the cabinet door, pass the door handle, and stop at the Snellen chart near the table lamp."
    )
    assert meta1["changed"] is True
    assert meta1["extra_terms"] == ["eye chart", "cabinet door", "door handle", "table lamp"]
    assert meta1["active_labels"][:240] == base_labels
    assert meta1["active_labels"][240:] == ["eye chart", "cabinet door", "door handle", "table lamp"]
    assert len(model.text_pe_calls) == 2
    assert len(model.set_classes_calls) == 2

    meta_repeat = configure(
        "Turn left at the cabinet door, pass the door handle, and stop between the eye test chart and the table lamp."
    )
    assert meta_repeat["changed"] is False
    assert len(model.text_pe_calls) == 2
    assert len(model.set_classes_calls) == 2

    model.next_class_id = 240
    for view_idx in range(4):
        rgb = np.zeros((6, 8, 3), dtype=np.uint8)
        rgb[:, :, 0] = 10 + view_idx
        rgb[:, :, 2] = 200 + view_idx
        out = segment(rgb)
        assert len(out) == 1
        assert out[0]["class_name"] == "eye chart"
        assert out[0]["canonical_label"] == "eye_chart"
        assert detector_matches_endpoint(out[0]["canonical_label"], "stop at the Snellen chart")

    assert len(model.predict_calls) == 4
    assert len(model.text_pe_calls) == 2
    assert len(model.set_classes_calls) == 2
    for idx, (bgr_img, kwargs) in enumerate(model.predict_calls):
        assert bgr_img[0, 0, 0] == 200 + idx
        assert bgr_img[0, 0, 2] == 10 + idx
        assert kwargs == {
            "conf": 0.48,
            "iou": 0.3,
            "agnostic_nms": True,
            "retina_masks": True,
            "verbose": False,
        }

    meta_reset = configure("stop at the eye chart", enabled=False)
    assert meta_reset["changed"] is True
    assert meta_reset["extra_terms"] == []
    assert meta_reset["active_labels"] == base_labels
    assert len(model.text_pe_calls) == 2
    assert len(model.set_classes_calls) == 3

    meta_re_enable = configure(
        "Turn left at the cabinet door, pass the door handle, and stop between the eye chart and the table lamp."
    )
    assert meta_re_enable["changed"] is True
    assert len(model.text_pe_calls) == 2
    assert len(model.set_classes_calls) == 4

    for idx in range(5):
        m = configure(f"stop at the custom{idx} chart", enabled=True, max_extra_terms=2)
        assert m["cache_size"] <= 4
    assert len(ns["_TEXT_PE_CACHE"]) <= 4

    many = ", ".join(f"stop at the custom{i} chart" for i in range(25))
    meta_capped = configure(many, enabled=True, max_extra_terms=100)
    assert len(meta_capped["extra_terms"]) == 16


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for fn in (
        test_canonical_label_and_target_identity_eye_chart_aliases_and_exclusions,
        test_bounded_instruction_terms_Compounds_and_route_exclusions,
        test_configure_task_vocabulary_cache_reset_and_four_view_predict,
    ):
        suite.addTest(unittest.FunctionTestCase(fn))
    return suite

