FROM continuumio/miniconda3

# Set the working directory
WORKDIR /workspace

# Install system dependencies for NetCDF (if needed)
RUN apt-get update && apt-get install -y \
    libnetcdf-dev \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python and ipykernel in base environment
RUN conda install -y python=3.11 ipykernel

# Clean up
RUN conda clean -afy

# Set the default command
CMD ["bash"]
