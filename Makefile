PYTHON ?= python
BASH ?= bash
ENV_FILE ?= $(if $(wildcard .env),.env,.env.example)
COMPOSE = docker compose --project-directory . --file docker-compose.yml --env-file $(ENV_FILE)

.PHONY: validate validate-deployment validate-evidence validate-models validate-model-structure validate-model-evidence architecture check-architecture compose-config up down status smoke smoke-models prepare-models clean-models capture-evidence

validate:
	$(PYTHON) scripts/validate_structure.py
	$(PYTHON) scripts/validate_module_map.py
	$(PYTHON) scripts/validate_deployment.py
	$(PYTHON) scripts/validate_runtime_evidence.py
	$(PYTHON) scripts/validate_model_repository.py --structure-only
	$(PYTHON) scripts/validate_model_evidence.py
	$(PYTHON) -m unittest discover -s tests/unit -p "test_*.py"

validate-deployment:
	$(PYTHON) scripts/validate_deployment.py

validate-evidence:
	$(PYTHON) scripts/validate_runtime_evidence.py

validate-models:
	$(PYTHON) scripts/validate_model_repository.py

validate-model-structure:
	$(PYTHON) scripts/validate_model_repository.py --structure-only

validate-model-evidence:
	$(PYTHON) scripts/validate_model_evidence.py

architecture:
	$(PYTHON) scripts/generate_dependency_graph.py

check-architecture:
	$(PYTHON) scripts/generate_dependency_graph.py --check

compose-config:
	$(COMPOSE) config --quiet

up:
	ML_DEV_OPS_ENV_FILE=$(ENV_FILE) $(BASH) deployment/scripts/run_environment.sh

down:
	ML_DEV_OPS_ENV_FILE=$(ENV_FILE) $(BASH) deployment/scripts/stop_environment.sh

status:
	ML_DEV_OPS_ENV_FILE=$(ENV_FILE) $(BASH) deployment/scripts/check_environment.sh

smoke:
	$(PYTHON) deployment/scripts/smoke_environment.py --env-file $(ENV_FILE)

prepare-models:
	$(PYTHON) scripts/model_preparation/prepare_models.py prepare

smoke-models:
	$(PYTHON) deployment/triton/smoke_models.py --env-file $(ENV_FILE)

clean-models:
	$(PYTHON) scripts/model_preparation/prepare_models.py clean

capture-evidence:
	$(PYTHON) deployment/scripts/capture_runtime_evidence.py
