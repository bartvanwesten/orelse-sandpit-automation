FROM continuumio/miniconda3

# Install system dependencies including Git LFS
RUN apt-get update && apt-get install -y \
    libnetcdf-dev \
    libhdf5-dev \
    curl \
    sudo \
    && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && apt-get install -y git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Create vscode user (expected by Codespaces)
RUN groupadd --gid 1000 vscode \
    && useradd --uid 1000 --gid vscode --shell /bin/bash --create-home vscode \
    && echo vscode ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/vscode \
    && chmod 0440 /etc/sudoers.d/vscode

# Install Python and Jupyter in base environment
RUN conda install -y python=3.11 ipykernel

# Initialize Git LFS globally
RUN git lfs install --system

# Clean up
RUN conda clean -afy

# Switch to vscode user
USER vscode

CMD ["bash"]
