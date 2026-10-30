.PHONY: install test lint smoke baselines ablations figures benchmark all

install:          ## editable install with dev + ONNX extras
	pip install -e ".[dev,deploy]"

test:             ## unit tests (~30 s on CPU)
	pytest

lint:
	ruff check pointcbm scripts tests

smoke:            ## 1 fold x 1 epoch end-to-end run + evaluation
	python scripts/train.py --config configs/cpu_ablation.yaml --folds 1 \
	  --set training.num_epochs=1 experiment.name=smoke experiment.save_dir=/tmp/pointcbm
	python scripts/evaluate.py --checkpoint_dir /tmp/pointcbm/smoke

baselines:        ## non-deep baselines (~1 min)
	python scripts/baseline_geometric.py --features geometric --out results/geometric_baseline
	python scripts/baseline_geometric.py --features openshape --out results/openshape_zeroshot

ablations:        ## controlled CPU ablations (~40 min on 2 cores)
	./experiments/run_cpu_ablations.sh

figures:          ## tables + figures from saved out-of-fold predictions
	python scripts/analyze_results.py --out docs/figures
	python scripts/visualize_attention.py --run results/lite_cbm_clip --fold 1 --out docs/figures

benchmark:        ## CPU latency + ONNX export
	python scripts/benchmark_inference.py --out results/inference_benchmark.json

all: test baselines ablations figures benchmark
