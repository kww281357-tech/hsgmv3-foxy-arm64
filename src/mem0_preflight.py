"""Check the required Mem0 backend before launching a navigation batch."""

import json
import logging
import os

from navigation_memory import NavigationMemory


def main():
    logging.basicConfig(level=logging.INFO)
    if os.environ.get("HSGM_MEMORY_BACKEND") != "mem0":
        raise RuntimeError("V3 comparison requires HSGM_MEMORY_BACKEND=mem0.")
    # Use a separate database so this check never supplies synthetic experience
    # to the navigation agent or imports old comparison memories.
    memory = NavigationMemory(storage_path=os.environ["HSGM_MEMORY_PATH"] + ".preflight")
    try:
        memory.search("navigation memory semantic retrieval readiness", limit=1)
        print("Mem0 preflight passed: " + json.dumps(memory.status(), ensure_ascii=False))
    finally:
        memory.close()


if __name__ == "__main__":
    main()

