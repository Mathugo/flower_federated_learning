FROM debian:buster-slim

# basics
RUN apt-get update
RUN apt-get install libopenblas-dev libopenmpi-dev python3-pip cmake -y

# update pip
RUN python3 -m pip install --upgrade pip

#RUN pip3 install Cython numpy

RUN mkdir /app
WORKDIR /app

## Installing Pytorch + Torchvision
RUN mkdir build
WORKDIR build
RUN apt-get install git bzip2 -y
RUN pip3 install scikit-build ninja

# PyTorch
RUN git clone https://github.com/pytorch/pytorch.git
WORKDIR pytorch
RUN git checkout v1.6.0 && git submodule update --init --recursive
ENV USE_NCCL=0 USE_QNNPACK=0 USE_PYTORCH_QNNPACK=0
RUN pip3 install -r requirements.txt
RUN python3 setup.py install

# torchvision
WORKDIR /app/build
RUN git clone https://github.com/pytorch/vision.git
# checkout v0.7.0 (the one compatible with PyTorch 1.6)
WORKDIR vision
RUN git checkout v0.7.0 && git submodule update --recursive
RUN apt-get install libavcodec-dev libavformat-dev libswscale-dev libjpeg8-dev zlib1g-dev -y
RUN python3 setup.py install

# update pip
RUN pip3 install --upgrade pip

# making sure the latest version of flower is installed
RUN pip3 install flwr==0.16.0

WORKDIR /app
RUN echo "done!"
