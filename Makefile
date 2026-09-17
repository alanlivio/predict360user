MAKEFLAGS += -s --no-print-directory

.PHONY: help deps test clean

help:
	@printf "%s\n" \
		"Usage: make [target]" \
		"" \
		"Targets:" \
		"  deps     Install dependencies" \
		"  test     Run tests" \
		"  clean    Clean build and temporary files"

deps:
	pip install -r requirements.txt
	pip install -e . --no-deps

test:
	pytest

clean:
	rm -rf build dist *.egg-info .pytest_cache .metaflow
