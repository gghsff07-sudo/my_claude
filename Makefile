.PHONY: setup test test-full train chat eval convert lint bench clean

# install editable with dev + HF deps (torch CPU wheel)
setup:
	python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	python -m pip install -e ".[hf,dev]"

# quick suite: everything that needs no artifacts
test:
	python -m pytest tests/ -q

# full suite: restore the fp32 checkpoint and build the HF artifact first
test-full:
	python scripts/restore_checkpoint.py
	python scripts/convert_to_hf.py --checkpoint checkpoints/best.pt
	python -m pytest tests/ -q

train:
	python -m fable_lm.train --steps 800 --run-name run2 --dialogue-mult 6 --data-mult 2

chat:
	python -m fable_lm.chat

eval:
	python -m fable_lm.eval

convert:
	python scripts/convert_to_hf.py
	python scripts/convert_tokenizer_to_hf.py

lint:
	python -m ruff check .

bench:
	python scripts/bench_generation.py

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -name __pycache__ -type d -exec rm -rf {} +
