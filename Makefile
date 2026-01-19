
.PHONY: quality style test

quality:
	black --check .
	isort --check-only .
	flake8 --max-line-length 119 --ignore=E203,W503 --exclude=.venv,venv,.env,env,build,dist .

style:
	black .
	isort .

test:
	pytest -sv ./src/

pip:
	rm -rf build/
	rm -rf dist/
	make style && make quality
	python -m build
	twine upload dist/* --verbose