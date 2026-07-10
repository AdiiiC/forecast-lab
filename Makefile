.PHONY: install test lint bench bench-cov bench-hier serve api web clean reproduce validate retrain

install:
	pip install -e .[dl,mlops,serving,dashboard,test]

test:
	PYTHONPATH=src pytest -q

lint:
	ruff check src tests

bench:
	PYTHONPATH=src python -m forecast_lab.cli --config configs/energy_v2.yaml --track

bench-cov:
	PYTHONPATH=src python -m forecast_lab.cli --config configs/energy_cov.yaml --track

bench-hier:
	PYTHONPATH=src python -m forecast_lab.cli_hier --config configs/retail_hier.yaml --track

# Reproduce any run from its saved config snapshot
#   make reproduce run=energy_v2
reproduce:
	@if [ -z "$(run)" ]; then echo "Usage: make reproduce run=<run_name>"; exit 1; fi
	@if [ ! -f runs/$(run)/config.yaml ]; then echo "No config snapshot at runs/$(run)/config.yaml"; exit 1; fi
	PYTHONPATH=src python -m forecast_lab.cli --config runs/$(run)/config.yaml

# Validate data against schema and check drift
#   make validate data=path/to/data.parquet  [ref=path/to/ref.parquet]
validate:
	@if [ -z "$(data)" ]; then echo "Usage: make validate data=<path> [ref=<path>]"; exit 1; fi
	PYTHONPATH=src python -c "from pipelines.prefect_flow import validate_data; import json; print(json.dumps(validate_data('$(data)', ref_path='$(ref)' or None), indent=2))"

# Check drift and retrain if needed
#   make retrain run=energy_v2 config=configs/energy_v2.yaml
retrain:
	@if [ -z "$(run)" ] || [ -z "$(config)" ]; then echo "Usage: make retrain run=<run> config=<cfg>"; exit 1; fi
	PYTHONPATH=src python -c "from pipelines.prefect_flow import retrain_flow; retrain_flow('runs/$(run)', 'runs/$(run)/ref_residuals.npy', '$(config)')"

serve:
	PYTHONPATH=src uvicorn forecast_lab.serving.app:app --host 0.0.0.0 --port 8000

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload

web:
	cd web && npm install && npm run dev

clean:
	rm -rf runs mlruns .pytest_cache __pycache__ */__pycache__ */*/__pycache__
