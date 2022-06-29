FROM mathugo/flower-base

RUN apt-get install wget -y

# Download and extract CIFAR-10
# To keep things simple, we keep this as part of the docker image.
# If the dataset is already in your system you can mount it instead.
ENV DATA_DIR=/app/data/cifar-10
RUN mkdir -p $DATA_DIR
WORKDIR $DATA_DIR
RUN wget https://www.cs.toronto.edu/\~kriz/cifar-10-python.tar.gz 
RUN tar -zxvf cifar-10-python.tar.gz

WORKDIR /app
# pip requirements
ADD requirements.txt .
RUN python3 -m pip install -r requirements.txt

# Scripts needed for Flower client
ADD src/client ./src/client
ADD src/utils ./src/utils
ADD src/models ./src/models
#ADD utils.py /app

ENTRYPOINT ["python3","-u","./main.py"]