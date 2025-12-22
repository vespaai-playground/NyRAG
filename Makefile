
.PHONY: quality style test

quality:
	black --check --line-length 119 --target-version py310 --exclude='/(venv|.venv|env|.env)/' .
	isort --check-only --skip venv --skip .venv --skip env --skip .env .
	flake8 --max-line-length 119 --exclude=venv,.venv,env,.env --ignore=E203,W503 .

style:
	black --line-length 119 --target-version py310 --exclude='/(venv|.venv|env|.env)/' .
	isort --skip venv --skip .venv --skip env --skip .env .

test:
	pytest -sv ./src/

pip:
	rm -rf build/
	rm -rf dist/
	make style && make quality
	python -m build
	twine upload dist/* --verbose