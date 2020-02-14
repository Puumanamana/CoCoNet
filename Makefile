test:
	pytest

coverage:
	pytest --cov=coconet

publish:
	pip install .
	python setup.py sdist
	twine upload dist/*

