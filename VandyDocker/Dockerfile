ARG GMSH_VERSION=4.8.0
ARG TINI_VERSION=v0.18.0

FROM iitrabhi/fenics as base

WORKDIR /home/fenics
COPY .bashrc /home/fenics

RUN sudo apt-get update
RUN sudo apt-get install -y libxinerama-dev
RUN sudo apt-get install -y libxcursor1
RUN sudo apt-get install -y libglu1-mesa


RUN pip3 install sklearn
RUN pip3 install lxml
RUN pip3 install meshx 
RUN pip3 install pandas
RUN pip3 install tabulate


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
