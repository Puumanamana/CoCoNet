test:
	pytest

coverage:
	pytest --cov=coconet

publish:
	pip install --ignore-installed .
	python setup.py sdist
	twine upload dist/*

