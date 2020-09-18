ARG GMSH_VERSION=4.3.0
ARG TINI_VERSION=v0.18.0

FROM quay.io/fenicsproject/stable:2019.1.0.r3 as base

WORKDIR /home/fenics
COPY .bashrc /home/fenics

ARG GMSH_VERSION
WORKDIR /tmp

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qq update && \
    apt-get -yq --with-new-pkgs -o Dpkg::Options::="--force-confold" upgrade && \
    apt-get -y install\
        libglu1 \
        libxcursor-dev \
        libxinerama1 && \
    apt-get -y install \
        python3-lxml && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# apt-get -y install texlive-latex-extra &&\
# Download Install gmsh
RUN cd /usr/local && \
    wget -nc --quiet http://gmsh.info/bin/Linux/gmsh-${GMSH_VERSION}-Linux64.tgz && \
    tar -xf gmsh-${GMSH_VERSION}-Linux64.tgz
ENV PATH=/usr/local/gmsh-${GMSH_VERSION}-Linux64/bin:$PATH

RUN pip install --upgrade pip
RUN pip install sphinx
# RUN pip3 install --no-cache-dir --no-binary=h5py h5py pygmsh
RUN pip3 install h5py
RUN pip3 install --no-cache-dir meshio==3.2.7

RUN pip install numba
RUN pip install scipy --upgrade
RUN pip install sklearn
WORKDIR /root

########################################

FROM base as notebook
LABEL description="DOLFIN-X Jupyter Notebook"
WORKDIR /root

ARG TINI_VERSION
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini && \
    pip3 install --no-cache-dir jupyter jupyterlab

ENTRYPOINT ["/tini", "--", "jupyter", "notebook", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]

########################################