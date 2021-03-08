VERSION ?= 0.9.0

test:
	pytest

coverage:
	pytest --cov=coconet

publish:
	pip install --ignore-installed .
	python setup.py sdist
	twine upload dist/*

container:
	sudo docker build --no-cache -f dockerfile -t nakor/coconet:$(VERSION) . \
	&& sudo docker push nakor/coconet:$(VERSION)
