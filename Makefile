PROJECT=gdec
PYTHON_SOURCE_DIRS=$(PROJECT) tests

all: format check test

check:
	@flake8 $(PYTHON_SOURCE_DIRS)
	@mypy $(PYTHON_SOURCE_DIRS)
	@isort --quiet --recursive --diff $(PYTHON_SOURCE_DIRS)
	@black --diff --check --quiet $(PYTHON_SOURCE_DIRS)

docs:
	@echo "Not implemented"

format:
	@isort --quiet --recursive $(PYTHON_SOURCE_DIRS)
	@black --quiet $(PYTHON_SOURCE_DIRS)

help:
	@echo "\tall:\tFormat, lint, and test"
	@echo "\tcheck:\tRun checks"
	@echo "\tformat:\tFormat code"
	@echo "\thelp:\tShow this help"
	@echo "\ttest:\tRun tests"

test:
	@pytest -q -s
