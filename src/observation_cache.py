"""Bounded, decision-local image work. No navigation or completion decisions."""
from collections import OrderedDict
from collections.abc import Sequence
import time


class LazyViews(Sequence):
    """Materialize side views only when a full planner/verifier consumes them."""
    def __init__(self, factories):
        self.factories = tuple(factories)
        self.images = {}

    def __len__(self):
        return len(self.factories)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index not in self.images:
            self.images[index] = self.factories[index]()
        return self.images[index]


class DecisionImageCache:
    """Own immutable PIL inputs until reset; separate full-size/thumbnail profiles.

    Callers must not mutate an image after handing it to this cache. Retaining
    the source object prevents Python id reuse from returning another image.
    The agent clears the cache at every decision and after observation updates.
    """
    def __init__(self, capacity=16):
        self.capacity = max(1, int(capacity))
        self.entries = OrderedDict()

    def encode(self, image, encoder, max_side=None):
        started = time.monotonic()
        key = (id(image), max_side)
        entry = self.entries.get(key)
        hit = entry is not None and entry[0] is image
        if hit:
            self.entries.move_to_end(key)
            encoded = entry[1]
        else:
            prepared = image
            if max_side is not None:
                prepared = image.copy()
                prepared.thumbnail((max_side, max_side))
            encoded = encoder(prepared)
            self.entries[key] = (image, encoded)
            while len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
        return encoded, hit, time.monotonic() - started

