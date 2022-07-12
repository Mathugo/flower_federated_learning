FROM pytorch/pytorch

ENV FLWR_USER=orange
ENV FLWR_PASS=orange123456789

WORKDIR /app/
ADD server/flower/requirements.txt .
RUN python3 -m pip install -r ./requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 git -y

#add src 
ADD server/flower/src /app/src
ADD server/flower/main.py /app/main.py
ADD server/flower/models /app/models
RUN mkdir /app/client
ADD client/src /app/client/src

ADD utils /app/utils
ADD models /app/models

