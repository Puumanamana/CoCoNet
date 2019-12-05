test:
	python tests/test_generators.py
	python tests/test_fragmentation.py

publish:
	pip install .
	python setup.py sdist
	twine upload dist/*
