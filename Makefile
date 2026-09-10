.PHONY: data clean

data:
	python build_pipeline.py --dataset all

clean:
	rm -rf data/processed/* data/features/*