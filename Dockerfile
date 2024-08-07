# Use an NVIDIA CUDA base image
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# Set the working directory in the container
WORKDIR /app

# Set environment variables
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_DATASETS_CACHE=/app/.cache/huggingface
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.7 \
    python3.7-venv \
    python3.7-distutils \
    python3-pip \
    python3-apt \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip for Python 3.7 specifically
RUN python3.7 -m pip install --no-cache-dir --upgrade pip
RUN ln -s /usr/bin/python3.7 /usr/bin/python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1


COPY requirements.txt /app/
COPY . /app
RUN pip3 install -e .
RUN pip3 install -r requirements.txt  --ignore-installed
RUN pip3 uninstall transformers -y
RUN apt-get update && apt-get install -y --no-install-recommends git
RUN pip3 install git+https://github.com/jordiclive/transformers.git@controlprefixes --ignore-installed
RUN pip3 install torchtext==0.8.0 torch==1.7.1

RUN apt-get update && apt-get install -y unzip && rm -rf /var/lib/apt/lists/*

RUN git lfs install

ENV DEBIAN_FRONTEND=

ENTRYPOINT ["tail"]
CMD ["-f","/dev/null"]
