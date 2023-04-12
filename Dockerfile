FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel
# FROM rocm/pytorch:latest

ADD . /workspace/resnet
WORKDIR /workspace/resnet

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
RUN apt-get install -y vim

RUN pip install --upgrade pip
RUN pip install --disable-pip-version-check -r requirements.txt