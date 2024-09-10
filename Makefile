.PHONY: install test tests lr features

# Detect the operating system
ifeq ($(OS),Windows_NT)
    PYTHON_PATH_SET := set PYTHONPATH=./ &&
else
    PYTHON_PATH_SET := PYTHONPATH=./
endif

default: test

install:
	pip install -e .
	pip install pytest-colored

test: tests

tests:
	$(PYTHON_PATH_SET) pytest -s --color=yes

lr:
	$(PYTHON_PATH_SET) pytest -s --color=yes ./tests/test_linear_regression.py

features:
	$(PYTHON_PATH_SET) pytest -s --color=yes ./tests/test_features.py