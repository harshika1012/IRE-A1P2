.PHONY: data features reranker nrms serving-benchmark extended-eval test clean

data:
	python build_pipeline.py --dataset all

features: data
	python build_behavioral_features.py --dataset all

reranker: features
	python run_reranker.py --dataset all

nrms: features
	python run_nrms.py --dataset all

serving-benchmark: reranker
	python run_serving_benchmark.py --dataset all

extended-eval: reranker
	python run_extended_eval.py --dataset all

test:
	pytest tests/ -q

clean:
	rm -rf data/processed/* data/features/*