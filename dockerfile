FROM continuumio/miniconda3:latest
LABEL author="carisdak@hawaii.edu"

RUN apt-get update && apt-get install -y locales procps git && apt-get clean -y
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && locale-gen

COPY conda_env.yml /
RUN conda env create -f /conda_env.yml && conda clean -a

#=================================================#
#================== ViralVerify ==================#
#=================================================#

RUN git clone https://github.com/ablab/viralVerify.git \
    && mv viralVerify/* /opt/conda/envs/virus_extraction/bin

#=================================================#
#==================== Diamond ====================#
#=================================================#

COPY diamond /opt/conda/envs/virus_extraction/bin

ENV PATH /opt/conda/envs/virus_extraction/bin:$PATH

ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8 
