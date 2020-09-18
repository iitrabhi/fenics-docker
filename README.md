# Installing FEniCS on Docker in MAC/Windows.

This repository contains the docker script containing all the dependencies that are necessary to carry out simulations using FEniCS at CMLab, Indian Institute of Technology Roorkee, India.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

## Installation on Linux

To install in Linux simply run the following commands in terminal:

```
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt-get update
sudo apt-get install --no-install-recommends fenics
```

These commands will just install FEniCS on your system and for all the other dependencies you have to do manual installation.

### Prerequisites

To follow along with the examples you need to install docker on your system. You need Windows 10 Education or Professional for this to work. This does not work on Windows 10 Home.

* [Docker](https://www.docker.com/products/docker-desktop)
* [CMDER](https://cmder.net/) (Only for Windows)

After installation open `cmder` and then go-to Settings(Win+Alt+P)➡import and choose the `cmlab.xml` provided in the repository.

Once the docker system in installed and running open CMDER/terminal and run:

- To install the command line interface of FEniCS run:

```
cd /path/to/this/repo
cd Docker
docker build --target base -t fenics .
```

- To install the Jyupter notebook interface run:

```
cd /path/to/this/repo
cd Docker
docker build --target notebook -t fenics_notebook .
```

Note that you should have the file named Dockerfile in the directory where you run docker build.

## Running

After building the docker image you can start the command line interface by running the following:

```
docker run -v host_system_path:/root/ -w /root/ -it fenics
```

To start the notebook use:

```
docker run -p 8888:8888 -v host_system_path:/root/ -w /root/ fenics_notebook
```

Note: you should replace the variable `host_system_path` with the path of the folder that contains your code. e.g. If  `D:\Codes` contains your code then to start the command line interface you have to run:

```
docker run -v D:\Codes:/root/ -w /root/ -it fenics
```

## Authors

* [Abhinav Gupta](https://computationalmechanics.in/rajib_teams/abhinav-gupta/)