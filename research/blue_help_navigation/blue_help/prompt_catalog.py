"""Load the hand-authored English review prompts; no model or network operations."""
from pathlib import Path

PROMPT_FILES = {
    "policy": "01_policy_en.txt",
    "question": "02_question_en.txt",
    "dummy_oracle": "03_dummy_oracle_answer_en.txt",
}


def load_prompts(language: str = "en") -> dict[str, str]:
    if language != "en":
        raise ValueError("This edition supports English prompts only")
    root = Path(__file__).resolve().parent / "prompt_data"
    return {role: (root / filename).read_text(encoding="utf-8")
            for role, filename in PROMPT_FILES.items()}
