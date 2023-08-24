FROM nvcr.io/nvidia/nvhpc:22.9-devel-cuda11.7-ubuntu22.04

RUN apt-get update && \
    apt-get -y install curl git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /venv

ARG CONDA_DIR=/venv/.miniconda3
ARG CONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN curl ${CONDA_URL} -o /tmp/miniconda3.sh && \
    bash /tmp/miniconda3.sh -b -u -p ${CONDA_DIR} && \
    ${CONDA_DIR}/bin/conda install -y cudatoolkit
ENV PATH=${CONDA_DIR}/bin:${PATH}

COPY requirements.txt requirements.txt
RUN ${CONDA_DIR}/bin/python3 -m pip install --no-cache-dir -r requirements.txt

WORKDIR /app
