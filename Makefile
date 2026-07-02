.PHONY: install test lint bench bench-cov bench-hier serve api web clean

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

serve:
    PYTHONPATH=src uvicorn forecast_lab.serving.app:app --host 0.0.0.0 --port 8000

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload

web:
	cd web && npm install && npm run dev

clean:
    rm -rf runs mlruns .pytest_cache __pycache__ */__pycache__ */*/__pycache__