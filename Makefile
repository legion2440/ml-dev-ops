PYTHON ?= python

.PHONY: validate architecture check-architecture

validate:
	$(PYTHON) scripts/validate_structure.py
	$(PYTHON) scripts/validate_module_map.py

architecture:
	$(PYTHON) scripts/generate_dependency_graph.py

check-architecture:
	$(PYTHON) scripts/generate_dependency_graph.py --check
