PY ?= python
RUN_ID ?= default
START ?=
END ?=

ARGS :=
ifneq ($(START),)
  ARGS += --start $(START)
endif
ifneq ($(END),)
  ARGS += --end $(END)
endif

.PHONY: help init run run-ml run-force run-ml-force eval eval-ml eval-force eval-ml-force

help:
	@echo "Targets: init, run, run-ml, run-force, run-ml-force, eval, eval-ml, eval-force, eval-ml-force"
	@echo "Vars: RUN_ID=default START=YYYY-MM-DD END=YYYY-MM-DD"

init:
	@$(PY) -c "from pathlib import Path; [Path(p).mkdir(parents=True, exist_ok=True) for p in ['data/raw','data/processed','data/experiments','models','references','reports/figures']]"

run:
	@$(PY) -m src.pipeline run --run-id $(RUN_ID) $(ARGS)

run-ml:
	@$(PY) -m src.pipeline run --run-id $(RUN_ID) --ml $(ARGS)

run-force:
	@$(PY) -m src.pipeline run --run-id $(RUN_ID) --force $(ARGS)

run-ml-force:
	@$(PY) -m src.pipeline run --run-id $(RUN_ID) --ml --force $(ARGS)

eval:
	@$(PY) -m src.pipeline eval --run-id $(RUN_ID) $(ARGS)

eval-ml:
	@$(PY) -m src.pipeline eval --run-id $(RUN_ID) --ml $(ARGS)

eval-force:
	@$(PY) -m src.pipeline eval --run-id $(RUN_ID) $(ARGS)

eval-ml-force:
	@$(PY) -m src.pipeline eval --run-id $(RUN_ID) --ml $(ARGS)
