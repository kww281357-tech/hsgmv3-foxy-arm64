.PHONY: test plan verify
PYTHON ?= python
test:
	$(PYTHON) -m unittest discover -s tests -v
plan:
	$(PYTHON) -m blue_help plan --batch-size 2
verify:
	$(PYTHON) scripts/verify_source.py
