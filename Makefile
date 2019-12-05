test:
	pytest tests/*.py

coverage:
	pytest --cov=coconet tests/*.py

publish:
	pip install .
	python setup.py sdist
	twine upload dist/*

