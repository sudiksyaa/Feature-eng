# .PHONY: install test tests lr features

# # Use Python to set PYTHONPATH
# PYTHON_PATH_SET := python -c "import os, sys; os.environ['PYTHONPATH'] = os.path.abspath('.')+os.pathsep+os.environ.get('PYTHONPATH',''); sys.exit(0)"

# default: test

# install:
# 	pip install -e .
# 	pip install pytest-colored

# test: tests

# tests:
# 	$(PYTHON_PATH_SET) && pytest -s --color=yes

# lr:
# 	$(PYTHON_PATH_SET) && pytest -s --color=yes ./tests/test_linear_regression.py

# features:
# 	$(PYTHON_PATH_SET) && pytest -s --color=yes ./tests/test_features.py


# .PHONY: install test

# default: test

# install:
# 	pip install -e .

# test:
# 	PYTHONPATH=./ pytest -s

# lr:
# 	PYTHONPATH=./ pytest -s ./tests/test_linear_regression.py
	
# features:
# 	PYTHONPATH=./ pytest -s ./tests/test_features.py


# Phony targets
.PHONY: all install test tests lr features

# Detect OS
ifeq ($(OS),Windows_NT)
    detected_OS := Windows
else
    detected_OS := $(shell uname -s)
endif

# Set path separator
ifeq ($(detected_OS),Windows)
	PATHSEP := ;
else
	PATHSEP := :
endif

# Set PYTHONPATH
export PYTHONPATH := .$(PATHSEP)$(PYTHONPATH)

# Default target
all: test

# Install dependencies
install:
	pip install -e .
	pip install pytest pytest-colored

# Run all tests
test: tests

tests:
	pytest -s --color=yes

# Run linear regression tests
lr:
	pytest -s --color=yes ./tests/test_linear_regression.py

# Run feature tests
features:
	pytest -s --color=yes ./tests/test_features.py