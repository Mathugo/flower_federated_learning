FROM pytorch/pytorch
#FROM debian:buster-slim

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 git python3-pip -y

WORKDIR /app/
ADD server/flower/requirements.txt .
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r ./requirements.txt

#add src 
ADD server/flower/src /app/src
ADD server/flower/main.py /app/main.py
RUN mkdir /app/client
ADD client/src /app/client/src
ADD utils /app/utils
ADD models /app/models


