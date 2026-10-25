"""Deterministic task-conditioned visual term extraction for open-vocabulary detection.

Extracts bounded noun phrases from navigation instructions without LLM calls or
detector imports, focusing on missing aliases (such as eye chart / Snellen chart),
named turn-at landmarks, and simple endpoint noun phrases.
"""
import re

from semantic_evidence import canonical_label


_EXACT_ALIASES = (
    (re.compile(r"\b(?:eye\s+test\s+charts?|snellen\s+charts?|eye\s+charts?)\b"), "eye chart"),
    (re.compile(r"\b(?:television\s+stands?|tv\s+stands?)\b"), "tv stand"),
    (re.compile(r"\b(?:bed\s*side\s+tables?|night\s+stands?)\b"), "nightstand"),
)
_SINGLE_SYNONYMS = {
    "couch": "sofa", "sofa": "sofa",
    "fridge": "refrigerator", "refrigerator": "refrigerator",
    "tv": "television", "television": "television",
    "stair": "stairs", "stairs": "stairs", "staircase": "stairs",
    "sculpture": "statue", "statue": "statue",
    "tub": "bathtub", "bathtub": "bathtub",
    "carpet": "rug", "mat": "rug", "rug": "rug",
    "armchair": "chair", "cupboard": "cabinet", "closet": "wardrobe",
}
_INVARIANT_PLURALS = {
    "stairs", "steps", "glasses", "sunglasses", "scissors", "clothes",
    "pants", "jeans", "shorts", "blinds", "curtains", "drawers",
}
_DETERMINERS = {
    "the", "a", "an", "your", "my", "our", "his", "her", "its", "their",
    "this", "that", "these", "those", "some", "any", "each", "every",
    "both", "all", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "first", "second", "third", "last", "next",
    "other", "another", "same",
}
_SIZE_STATE_MODIFIERS = {
    "small", "large", "big", "little", "huge", "tiny", "tall", "short",
    "long", "wide", "narrow", "high", "low", "old", "new", "open",
    "closed", "nearest", "closest", "far", "farthest", "furthest",
    "main", "final", "target",
}
_SURFACE_MODIFIERS = {
    "white", "black", "red", "blue", "green", "yellow", "brown", "gray",
    "grey", "silver", "gold", "golden", "orange", "purple", "pink",
    "beige", "tan", "dark", "light", "bright", "wooden", "wood",
    "metal", "glass", "plastic", "leather", "stone", "marble", "round",
    "square", "rectangular",
}
_ROUTE_VERBS = {
    "walk", "walking", "go", "going", "move", "moving", "head", "heading",
    "proceed", "proceeding", "continue", "continuing", "turn", "turning",
    "stop", "stopping", "wait", "waiting", "stand", "standing", "climb",
    "climbing", "descend", "descending", "enter", "entering", "exit",
    "exiting", "leave", "leaving", "pass", "passing", "cross", "crossing",
    "follow", "following", "take", "taking", "make", "making", "reach",
    "reaching", "arrive", "arriving", "look", "looking", "face", "facing",
    "approach", "approaching",
}
_NON_OBJECT_HEADS = {
    "wall", "floor", "ceiling", "room", "bedroom", "bathroom", "kitchen",
    "living", "dining", "office", "hall", "hallway", "corridor", "walkway",
    "passage", "passageway", "entry", "entryway", "entrance", "foyer",
    "lobby", "landing", "stairwell", "area", "space", "zone", "section",
    "corner", "intersection", "way", "path", "route", "direction",
    "distance", "left", "right", "straight", "forward", "backward",
    "back", "front", "side", "top", "bottom", "end", "middle", "center",
    "edge", "foot", "base", "point", "spot", "place", "location",
    "position", "level", "story", "building", "house", "home",
    "apartment", "garage", "basement", "attic", "patio", "porch",
    "balcony", "deck", "yard", "garden", "outside", "inside", "there",
    "here", "it", "them", "you", "one", "ones", "thing", "things",
    "object", "objects", "step", "turn", "view", "scene", "image",
}
_PREFIX = re.compile(
    r"^\s*(?:(?:next\s+to|in\s+front\s+of|on\s+top\s+of|out\s+of|to\s+the\s+(?:left|right)\s+of|"
    r"at\s+the\s+(?:end|top|bottom|side|front|back)\s+of)|"
    r"(?:after|before)\s+(?:stepping|crossing|passing|walking|going|moving|turning)|"
    r"(?:(?:left|right)\s+)?(?:side|top|bottom|end|foot|base|edge|back|middle|center|front)\s+of|"
    r"(?:walk|go|move|head|proceed|continue|stop|wait|stand|standing|end|finish|halt|pause|"
    r"reach|arrive|approach|face|facing|look|looking|you|should|now|then|please|and|"
    r"the|a|an|your|to|in|on|at|by|beside|near|under|above|below|behind|through|into|"
    r"toward|towards|past|along|down|up|front|left|right|around))\b\s*"
)
_RELATION = re.compile(
    r"[.;,\n]|\b(?:next\s+to|in\s+front\s+of|on\s+top\s+of|out\s+of|across\s+from|"
    r"to|of|in|on|out|above|below|beside|near|with|at|by|behind|under|underneath|"
    r"beneath|against|before|after|between|from|into|onto|through|toward|towards|"
    r"past|along|across|over|and|or|that|which|where|when|while|until|then|is|are|was|were)\b"
)
_CLAUSE_SPLIT = re.compile(
    r"[.;,\n]+|\b(?:then|and)\b(?=\s+(?:you\s+|please\s+|now\s+)?(?:should\s+)?"
    r"(?:turn|make|take|stop|wait|stand|end|finish|halt|pause|reach|arrive|"
    r"walk|go|move|head|proceed|continue|climb|descend|enter|exit|leave|pass|cross|"
    r"follow|approach|face|look)\b)"
)
_TURN_AT = re.compile(
    r"\b(?:turn(?:ing)?(?:\s+(?:left|right|around|sharply|slightly))?|"
    r"make\s+a\s+(?:left|right)(?:\s+turn)?|take\s+a\s+(?:left|right)(?:\s+turn)?)\s+"
    r"(?:at|by|near|before|after|past|beside|next\s+to|in\s+front\s+of|toward|towards)\s+(.+)"
)
_PRE_TURN_AT = re.compile(
    r"^\s*(?:at|by|near|before|after|past)\s+(.+?)\s+(?:turn(?:ing)?|make\s+a\s+(?:left|right)|take\s+a\s+(?:left|right))\b"
)
_ENDPOINT = re.compile(
    r"\b(?:(?:stop(?:ping)?|wait(?:ing)?|stand(?:ing)?|end(?:ing)?|finish(?:ing)?|"
    r"halt(?:ing)?|pause|pausing|reach(?:ing)?|arriv(?:e|ing))|"
    r"(?:walk|go|move|head|proceed|continue|step)(?:ing)?\s+(?:(?:straight|forward|ahead)\s+)?"
    r"(?:up\s+to|over\s+to|to|toward|towards|past|through|by|around|over|under|across|along|beside|near|between|into|out\s+of|from|down|up)|"
    r"pass(?:ing)?(?:\s+(?:by|through|over|under|around|beside|near|past))?|"
    r"(?:enter|entering|exit|exiting|leave|leaving|cross|crossing|climb|climbing|descend|descending|follow|following)"
    r"(?:\s+(?:through|into|out\s+of|from|up|down|over|across|along|past|by))?|"
    r"approach(?:ing)?|face|facing|look(?:ing)?\s+at)\b\s+(.+)"
)


def _singularize_word(word: str) -> str:
    if word in _INVARIANT_PLURALS or len(word) <= 2:
        return word
    if word == "shelves":
        return "shelf"
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "xes", "sses")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def normalize_visual_term(term: str) -> str:
    text = re.sub(r"[\s_-]+", " ", str(term or "").strip().lower())
    for pattern, replacement in _EXACT_ALIASES:
        text = pattern.sub(replacement, text)
    words = re.findall(r"[a-z0-9]+", text)
    if not words:
        return ""
    words[-1] = _singularize_word(words[-1])
    joined = " ".join(words)
    for pattern, replacement in _EXACT_ALIASES:
        joined = pattern.sub(replacement, joined)
    return joined


def _without_prefix(text: str) -> str:
    while match := _PREFIX.match(text):
        text = text[match.end():]
    return text


def _clean_np(raw: str):
    text = normalize_visual_term(raw) if re.fullmatch(r"\s*(?:eye\s+test\s+charts?|snellen\s+charts?|eye\s+charts?)\s*", str(raw or ""), re.I) else str(raw or "").lower()
    for pattern, replacement in _EXACT_ALIASES:
        text = pattern.sub(replacement, text)
    text = _without_prefix(text)
    phrase = _RELATION.split(text, maxsplit=1)[0]
    words = re.findall(r"[a-z0-9]+", phrase)
    while words and words[0] in _DETERMINERS:
        words.pop(0)
    while len(words) > 1 and words[0] in _SIZE_STATE_MODIFIERS:
        words.pop(0)
    while len(words) >= 3 and words[0] in _SURFACE_MODIFIERS:
        words.pop(0)
    if not words or len(words) > 3:
        return None
    if any(word in _ROUTE_VERBS for word in words):
        return None
    normalized = normalize_visual_term(" ".join(words))
    norm_words = normalized.split()
    if not norm_words or norm_words[-1] in _NON_OBJECT_HEADS:
        return None
    return normalized


def _extract_phrase_targets(raw: str) -> list:
    text = re.sub(r"[_-]+", " ", str(raw or "").strip().lower())
    for pattern, replacement in _EXACT_ALIASES:
        text = pattern.sub(replacement, text)
    trimmed = _without_prefix(text)
    if not trimmed:
        return []
    items = []
    for segment in _RELATION.split(trimmed):
        cleaned = _clean_np(segment)
        if cleaned:
            items.append(cleaned)
    return items


def is_equivalent_label(term: str, existing_labels) -> bool:
    norm = normalize_visual_term(term)
    if not norm:
        return True
    canon = canonical_label(norm)
    existing_canon = {canonical_label(normalize_visual_term(item)) for item in existing_labels if item}
    if canon in existing_canon:
        return True
    if "_" not in canon and canon in _SINGLE_SYNONYMS:
        target = _SINGLE_SYNONYMS[canon]
        return any(_SINGLE_SYNONYMS.get(item, item) == target for item in existing_canon if "_" not in item)
    return False


def bounded_instruction_terms(instruction, max_extra_terms: int = 16) -> list:
    limit = min(16, max(0, int(max_extra_terms)))
    if limit == 0 or not instruction:
        return []
    text = re.sub(r"[_-]+", " ", str(instruction).strip().lower())
    raw_candidates = []
    if re.search(r"\b(?:eye\s+test\s+charts?|snellen\s+charts?|eye\s+charts?)\b", text):
        raw_candidates.append("eye chart")
    clauses = [c.strip() for c in _CLAUSE_SPLIT.split(text) if c and c.strip()]
    has_route_verbs = any(word in _ROUTE_VERBS for word in re.findall(r"[a-z0-9]+", text))
    for clause in clauses:
        for turn_match in _TURN_AT.finditer(clause):
            raw_candidates.append(turn_match.group(1))
        pre_turn = _PRE_TURN_AT.match(clause)
        if pre_turn:
            raw_candidates.append(pre_turn.group(1))
        for end_match in _ENDPOINT.finditer(clause):
            raw_candidates.append(end_match.group(1))
        if not has_route_verbs:
            raw_candidates.append(clause)
    results = []
    seen = set()
    for raw in raw_candidates:
        for term in _extract_phrase_targets(raw):
            key = canonical_label(term)
            if key and key not in seen:
                seen.add(key)
                results.append(term)
                if len(results) >= limit:
                    return results
    return results

