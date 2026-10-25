import json
from pathlib import Path
import re
import unittest

from blue_help.help_offer import HelpOfferMachine, OFFER_TEXT
from blue_help.prompt_catalog import load_prompts, PROMPT_FILES

ROOT = Path(__file__).resolve().parents[1]
IDS = [40, 41, 52, 53, 237, 249, 254, 301, 387, 630, 712, 715, 744, 822, 894, 1334, 1354, 1612, 1720, 1755]


class EnglishEditionTests(unittest.TestCase):
    def test_three_complete_english_files_readable(self):
        prompts = load_prompts()
        self.assertEqual(set(prompts), {"policy", "question", "dummy_oracle"})
        for role, text in prompts.items():
            with self.subTest(role=role):
                self.assertGreater(len(text), 2000)
                self.assertFalse(re.search(r"[\u4e00-\u9fff]", text))
                self.assertIn("English", text)
                self.assertRegex(text, r"1[- ]meter")
                self.assertRegex(text, r"foot[- ]center")
                self.assertIn("same floor", text.lower())
                self.assertRegex(text, r"line of sight|LOS|unobstructed")
                if role != "question": self.assertIn(OFFER_TEXT, text)

    def test_no_chinese_prompt_fallback(self):
        with self.assertRaises(ValueError):
            load_prompts("zh")
        self.assertFalse(list((ROOT / "prompts").glob("*_zh.txt")))

    def test_config_selects_only_english_files(self):
        config = json.loads((ROOT / "configs/config.review.json").read_text())
        self.assertEqual(config["runtime_language"], "en")
        self.assertEqual(config["prompt_language"], "en")
        self.assertEqual(config["question_text_field"], "question_en")
        self.assertEqual(config["answer_text_field"], "answer_en")
        self.assertEqual({Path(p).name for p in config["prompt_files"]}, set(PROMPT_FILES.values()))
        self.assertEqual(config["interaction_radius_m"], 1.0)
        self.assertEqual(config["max_successfully_delivered_answers_per_episode"], 3)

    def test_private_twenty_scene_review_cards_are_not_bundled(self):
        self.assertFalse((ROOT / "docs/archive/MANUAL_20_EPISODE_REVIEW_EN.md").exists())
        from blue_help.data import FROZEN_IDS
        self.assertEqual(FROZEN_IDS, IDS)

    def test_english_question_and_answer_examples(self):
        for path in sorted((ROOT / "examples").glob("*.json")):
            text = path.read_text()
            self.assertFalse(re.search(r"[\u4e00-\u9fff]", text))
            self.assertNotIn("question_zh", text)
            self.assertNotIn("answer_zh", text)
            self.assertIsInstance(json.loads(text), dict)
        question = json.loads((ROOT / "examples/02_question.json").read_text())
        answer = json.loads((ROOT / "examples/03_answer.json").read_text())
        self.assertTrue(question["question_en"])
        self.assertTrue(answer["answer_en"])
        self.assertNotIn("goal_coordinates", answer)

    def test_offer_runtime_text_is_english(self):
        machine = HelpOfferMachine()
        offer = machine.observe([{"dummy_id": "synthetic_blue", "horizontal_distance_m": .8,
                                 "spatial_eligible": True, "interaction_radius_m": 1.0}], step=0)
        self.assertEqual(offer["text"], OFFER_TEXT)
        self.assertTrue(offer["text"].isascii())
        self.assertIn("Do you need any help?", offer["text"])
        self.assertEqual(machine.snapshot()["delivered_answers"], 0)

    def test_english_answer_delivered_once(self):
        machine = HelpOfferMachine()
        offer = machine.observe([{"dummy_id": "synthetic_blue", "horizontal_distance_m": .8,
                                 "spatial_eligible": True, "interaction_radius_m": 1.0}], step=0)
        request = machine.choose(offer["offer_id"], "accept", step=0, spatial_eligible_now=True)
        answer = json.loads((ROOT / "examples/03_answer.json").read_text())["answer_en"]
        delivered = machine.deliver(request["request_id"], answer, step=1, answer_validated=True,
                                    spatial_eligible_now=True, pose_reference_valid=True)
        self.assertEqual(delivered["answer_text"], answer)
        self.assertEqual(machine.snapshot()["delivered_answers"], 1)


if __name__ == "__main__":
    unittest.main()

class PackagedPromptTests(unittest.TestCase):
    def test_one_packaged_prompt_source_of_truth(self):
        prompts = load_prompts()
        root=Path(__file__).resolve().parents[1]/'blue_help/prompt_data'
        for role, name in PROMPT_FILES.items():
            self.assertEqual(prompts[role], (root/name).read_text())
