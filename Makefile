PY ?= .venv/bin/python

.PHONY: venv install test demo bench bench-sim charts security clean

venv:
	/opt/homebrew/opt/python@3.14/bin/python3.14 -m venv .venv || python3 -m venv .venv

install: venv
	$(PY) -m pip install -U pip wheel
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) tests/test_gateway.py

demo:
	$(PY) demo.py "reboot EC2 instance i-0abc123" --llm

security:
	$(PY) scripts/security_demo.py

bench:
	$(PY) benchmark/run_benchmark.py

bench-sim:
	$(PY) benchmark/run_benchmark.py --provider simulated

charts:
	$(PY) benchmark/make_charts.py

clean:
	rm -rf data/audit.sqlite3 data/desc_hashes.json data/real_tools_cache.json
	rm -f benchmark/results/*.csv benchmark/results/*.json benchmark/results/*.png
