ARG GMSH_VERSION=4.8.0
ARG TINI_VERSION=v0.18.0

FROM quay.io/fenicsproject/stable:2019.1.0.r3 as base

WORKDIR /home/fenics
COPY .bashrc /home/fenics

ARG GMSH_VERSION
WORKDIR /tmp

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

RUN pip install --upgrade pip
RUN pip3 install sphinx

RUN pip3 install pygmsh==5.0.0
RUN pip3 install --no-cache-dir --no-binary=h5py h5py
RUN pip3 install --no-cache-dir meshio==3.2.7

RUN pip3 install numba
RUN pip3 install scipy --upgrade
RUN pip3 install sklearn
RUN pip3 install lxml
RUN pip3 install meshx 
RUN pip3 install --user gmsh




WORKDIR /root

########################################

FROM base as notebook
LABEL description="DOLFIN-X Jupyter Notebook"
WORKDIR /root

ARG TINI_VERSION
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini && \
    pip3 install --upgrade --no-cache-dir jupyter jupyterlab


EXPOSE 8888/tcp
ENV SHELL /bin/bash

ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
########################################
