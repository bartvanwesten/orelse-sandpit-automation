FROM continuumio/miniconda3

# Install system dependencies including Git LFS
RUN apt-get update && apt-get install -y \
    libnetcdf-dev \
    libhdf5-dev \
    curl \
    && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && apt-get install -y git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Install Python and Jupyter in base environment
RUN conda install -y python=3.11 ipykernel

# Initialize Git LFS globally
RUN git lfs install --system

# Clean up
RUN conda clean -afy

CMD ["bash"]
