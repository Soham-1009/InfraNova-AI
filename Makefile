.PHONY: download preprocess train evaluate pipeline lint clean

download:
	python scripts/download/download_landsat9.py

preprocess:
	python scripts/preprocessing/process_landsat_patches.py
	python scripts/preprocessing/split_patches.py

train:
	python scripts/training/run_ablation_study.py

evaluate:
	python scripts/evaluation/evaluate.py

pipeline:
	python scripts/pipeline/run_pipeline.py --stage all

lint:
	pre-commit run --all-files

clean:
	rm -rf __pycache__ .pytest_cache outputs/models/*
