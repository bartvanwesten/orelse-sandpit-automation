FROM mcr.microsoft.com/devcontainers/python:3.11

# Install system dependencies for NetCDF and geospatial tools
RUN apt-get update && apt-get install -y \
    libnetcdf-dev \
    libhdf5-dev \
    netcdf-bin \
    libgeos-dev \
    libproj-dev \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Install conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/miniconda \
    && rm miniconda.sh \
    && /opt/miniconda/bin/conda init bash

ENV PATH="/opt/miniconda/bin:$PATH"
