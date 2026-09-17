MAKEFLAGS += -s --no-print-directory

export PYTHONPATH := .:$(PYTHONPATH)
ARGS ?= --dataset david --epochs 2

.PHONY: help deps test run eval-baselines eval-all clean

help:
	@printf "%s\n" \
		"Usage: make [target]" \
		"" \
		"Targets:" \
		"  deps           Install dependencies" \
		"  test           Run tests" \
		"  run            Benchmark baseline models (default: eval-baselines)" \
		"  eval-baselines Benchmark baselines, pos_only, and pos_only_3d" \
		"  eval-all       Benchmark and compare all models" \
		"  clean          Clean build and temporary files"

deps:
	pip install -r requirements.txt
	pip install -e . --no-deps

test:
	pytest

run: eval-baselines

eval-baselines:
	python3 predict360user/pipelines/benchmark_flow.py run --models "no_motion,interpolation,pos_only,pos_only_3d" $(ARGS)

eval-all:
	python3 predict360user/pipelines/benchmark_flow.py run --models "all" $(ARGS)

clean:
	rm -rf build dist *.egg-info .pytest_cache .metaflow
