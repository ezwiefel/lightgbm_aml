# Modified from Dockerfile found at https://github.com/microsoft/LightGBM/blob/master/docker/dockerfile-cli (commit 83ecb38)
FROM mcr.microsoft.com/azureml/base:openmpi3.1.2-ubuntu16.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        build-essential \
        gcc \
        g++ \
        git && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --recursive --branch stable --depth 1 https://github.com/Microsoft/LightGBM && \
    mkdir LightGBM/build && \
    cd LightGBM/build && \
    cmake .. && \
    make -j $(nproc) && \
    make install && \
    cd ../.. && \
    rm -rf LightGBM