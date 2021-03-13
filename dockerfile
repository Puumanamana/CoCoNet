FROM pytorch/pytorch:1.8.0
LABEL author="carisdak@hawaii.edu"

RUN apt-get update \
    && apt-get install -y --no-install-recommends git g++ procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY conda.yaml .

RUN conda env update -n base --file conda.yaml

WORKDIR /workspace

RUN git clone https://github.com/Puumanamana/CoCoNet.git \
    && pip install /workspace/CoCoNet/ \
    && rm -rf CoCoNet
