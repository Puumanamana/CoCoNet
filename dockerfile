FROM continuumio/miniconda3:4.8.2
LABEL author="carisdak@hawaii.edu"

RUN apt-get update \
    && apt-get install -y --no-install-recommends git g++ procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY conda.yaml .

RUN conda env update -n base --file conda.yaml

RUN git clone https://github.com/Puumanamana/CoCoNet.git \
    && cd CoCoNet \
    && pip install . \
    && rm -rf CoCoNet
