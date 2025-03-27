ARG GMSH_VERSION=4.13.0
ARG TINI_VERSION=v0.18.0

# Use the official Ubuntu base image
FROM ubuntu:latest AS base


WORKDIR /home/fenics
COPY .bashrc /home/fenics

ARG GMSH_VERSION
WORKDIR /tmp

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list and install necessary dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:fenics-packages/fenics \
    && apt-get update \
    && apt-get install -y fenics \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install git
RUN apt-get update && apt-get install -y git && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y python3-setuptools && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y wget && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y python3 && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y python3-pip && apt-get clean && rm -rf /var/lib/apt/lists/*
# RUN apt install texlive-fonts-recommended texlive-fonts-extra

# Clone FEniCS-shells 

RUN git clone https://github.com/umeenukrishnan/fenics-shell.git && \
    cd fenics-shell && \
    python3 setup.py install

# Install TIGAR

RUN git clone https://github.com/david-kamensky/tIGAr.git && \
    cd tIGAr && \
    python3 setup.py install


# Download Install gmsh
RUN cd /usr/local && \
    wget -nc --quiet http://gmsh.info/bin/Linux/gmsh-${GMSH_VERSION}-Linux64.tgz && \
    tar -xf gmsh-${GMSH_VERSION}-Linux64.tgz
ENV PATH=/usr/local/gmsh-${GMSH_VERSION}-Linux64/bin:$PATH


# RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y \
    python3-sphinx \
    python3-pygmsh \
    python3-h5py \
    python3-meshio \
    python3-pyvista \
    # python3-numba \
    python3-scipy \
    python3-sklearn \
    python3-lxml \
    python3-pandas \
    python3-tabulate \
    python3-svglib && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


WORKDIR /root

# ########################################

# FROM base as notebook
# LABEL description="DOLFIN-X Jupyter Notebook"
# WORKDIR /root

# ARG TINI_VERSION
# ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
# RUN chmod +x /tini && \
#     pip3 install --upgrade --no-cache-dir jupyter jupyterlab


# EXPOSE 8888/tcp
# ENV SHELL /bin/bash

# ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
# ########################################
