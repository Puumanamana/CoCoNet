VERSION ?= 1.1.0

test:
	coconet run --fasta tests/sim_data/assembly.fasta --bam tests/sim_data/*.bam --output output_test --threads 1 \
		--n-train 64 --n-test 8 --batch-size 2 --min-prevalence 0 --test-ratio 0.2 --n-frags 5 \
		--compo-neurons 8 4 --cover-neurons 8 4  --cover-kernel 2 --wsize 2 --wstep 2

coverage:
	pytest --cov=coconet

publish:
	pytest \
	  && python setup.py sdist \
	  && twine upload dist/*

container:
	sudo docker build --no-cache -f dockerfile -t nakor/coconet:$(VERSION) . \
	&& sudo docker push nakor/coconet:$(VERSION)
