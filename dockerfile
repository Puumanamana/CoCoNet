FROM continuumio/miniconda3:latest
LABEL author="carisdak@hawaii.edu"

RUN apt-get update && apt-get install -y git

RUN git clone https://github.com/Puumanamana/CoCoNet.git \
    && cd CoCoNet \
    && pip install . \
    && rm -rf CoCoNet


