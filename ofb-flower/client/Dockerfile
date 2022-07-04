FROM mathugo/flower-base

RUN apt-get install wget -y

# don't forget to mount data
WORKDIR /app
# pip requirements
ADD requirements.txt .
RUN python3 -m pip install -r requirements.txt

# Scripts needed for Flower client
ADD client/src/client ./src/client
ADD client/src/utils ./src/utils
ADD client/src/models ./src/models
ADD client/src/pipeline ./src/pipeline
ADD client/src/app ./src/app
ADD client/main.py ./main.py
ADD utils ./src/utils
ADD models ./src/models

#ENTRYPOINT ["python3","-u","./main.py"]