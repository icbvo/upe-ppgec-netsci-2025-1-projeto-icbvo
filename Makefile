# Makefile for GNN link prediction project

PYTHON := python
ENV_NAME := linkpred-gnn

.PHONY: help env install train clean

help:
	@echo "Targets:"
	@echo "  make env       - create conda environment (environment.yml)"
	@echo "  make install   - pip install package in editable mode"
	@echo "  make train     - run link prediction training script"
	@echo "  make clean     - remove __pycache__ and temporary files"

env:
	conda env create -f environment.yml || conda env update -f environment.yml

install:
	$(PYTHON) -m pip install -e .

train:
	cd gnn && $(PYTHON) train_link_prediction_gnn.py

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + || true
	find . -name "*.pyc" -delete || true
