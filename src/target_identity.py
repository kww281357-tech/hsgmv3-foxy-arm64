"""Conservative object-category binding for endpoint detector evidence.

This resolves named object classes, not visual instance identity. Ambiguous or
uncovered nouns remain unbound and must use independently confirmed RGB boxes.
"""
import re


_GROUPS = (
    ('sofa', 'couch'), ('refrigerator', 'fridge'), ('television', 'tv'),
    ('stairs', 'stair', 'staircase', 'steps'), ('door', 'doorway', 'archway', 'doorframe'),
    ('statue', 'sculpture'), ('wardrobe', 'closet'), ('cabinet', 'cupboard', 'curio'),
    ('dresser', 'chest'), ('rug', 'carpet', 'mat'), ('bathtub', 'tub', 'hottub'),
    ('chair', 'armchair'), ('stool', 'barstool'), ('railing', 'banister', 'handrail'),
    ('painting', 'picture', 'artwork'), ('pillar', 'column'),
    ('lamp', 'light', 'chandelier'),
    ('counter', 'countertop'),
    ('bar', 'workbench'),
    ('nightstand', 'endtable'), ('shelf', 'bookshelf', 'shelves'),
)
_SINGLE = ('bed desk table counter stove oven microwave sink toilet mirror '
           'window vase bench ottoman plant clock piano violin guitar thermostat altar '
           'treadmill island tvstand eyechart').split()
_CATEGORY = {word: group[0] for group in _GROUPS for word in group}
_CATEGORY.update({word: word for word in _SINGLE})
_PHRASES = {
    'bedside table': 'nightstand', 'bed side table': 'nightstand', 'night stand': 'nightstand',
    'end table': 'endtable', 'tv stand': 'tvstand', 'television stand': 'tvstand',
    'door frame': 'doorframe', 'door jamb': 'doorframe', 'picture frame': 'picture',
    'door handle': 'doorhandle', 'door knob': 'doorknob', 'door stop': 'doorstop',
    'cabinet door': 'cabinetdoor',
    'book shelf': 'bookshelf', 'bar stool': 'barstool', 'hot tub': 'hottub',
    'eye chart': 'eyechart', 'eye test chart': 'eyechart', 'snellen chart': 'eyechart',
}
_RELATION = re.compile(r'[.;,\n]|\b(?:to|of|in|on|out|above|below|beside|near|with|at|by|behind|'
                       r'before|after|between|and|that|which|where|then|is|are|was|were)\b')
_PREFIX = re.compile(
    r'^\s*(?:(?:next\s+to|in\s+front\s+of|out\s+of)|'
    r'(?:after|before)\s+(?:stepping|crossing|passing|walking|going|moving)|'
    r'(?:(?:left|right)\s+)?(?:side|top|bottom|end|foot|base|edge|back|middle)\s+of|'
    r'(?:walk|go|move|head|proceed|continue|stop|wait|stand|standing|you|should|now|then|please|'
    r'straight|forward|ahead|until|till|get|getting|reach|reaching|arrive|arriving|'
    r'the|a|an|your|to|in|on|at|by|beside|near|under|above|below|behind|through|into|'
    r'toward|towards|past|along|down|up|front))\b\s*')


def _words(text):
    text = re.sub(r'[_-]+', ' ', str(text).lower())
    for phrase, word in sorted(_PHRASES.items(), key=lambda item: -len(item[0])):
        text = re.sub(r'\b' + re.escape(phrase) + r's?\b', word, text)
    return text


def _category(word):
    if word in _CATEGORY:
        return _CATEGORY[word]
    singular = word[:-3] + 'y' if word.endswith('ies') else word[:-2] if word.endswith(('ches', 'shes', 'xes', 'sses')) else word[:-1] if word.endswith('s') else word
    return _CATEGORY.get(singular)


def _without_navigation_prefix(text):
    while match := _PREFIX.match(text):
        text = text[match.end():]
    return text


def object_category(text):
    """Use the head object noun, excluding spatial/descriptive landmarks.

    Compound heads matter: a bathroom cabinet is a cabinet, a table lamp is a
    lamp, and a TV stand is furniture rather than a television detection.
    """
    text = _without_navigation_prefix(_words(text))
    # Stop at the first relation even if its primary noun is unsupported. A
    # wall beside a sofa must not become a sofa target by skipping "wall".
    phrase = _RELATION.split(text, maxsplit=1)[0]
    words = re.findall(r'[a-z0-9]+', phrase)
    return _category(words[-1]) if words else None


def _endpoint_phrase(text):
    text = _words(text)
    # Prefer a named final arrival over earlier transit objects. Pronoun-only
    # endings retain the preceding noun for the existing endpoint-clause logic.
    arrivals = list(re.finditer(r'\b(?:stop|wait|stand|standing)\b', text))
    if arrivals and object_category(text[arrivals[-1].end():]):
        text = text[arrivals[-1].end():]
    return _without_navigation_prefix(text)


def is_between_endpoint(text):
    return bool(re.match(r'between\b', _endpoint_phrase(text)))


def endpoint_object_roles(text):
    """Return primary/reference categories for the localized endpoint relation."""
    text = _endpoint_phrase(text)
    # Only a leading between relation defines the goal pair. "The door between
    # two paintings" still targets a door, with paintings as landmarks.
    between = re.match(r'between\b(.+?)\band\b(.+)', text)
    if between:
        return {'primary': object_category(between.group(1)),
                'reference': object_category(between.group(2))}
    if re.match(r'between\b', text):
        category = object_category(text.split('between', 1)[1])
        return {'primary': category, 'reference': category}
    primary = object_category(text)
    # Doorway reference_region is the other jamb, not a nearby statue/landmark.
    return {'primary': primary, 'reference': primary if primary == 'door' else None}


def detector_matches_endpoint(label, text, role='primary'):
    expected = endpoint_object_roles(text).get(role)
    return expected is not None and object_category(label) == expected

