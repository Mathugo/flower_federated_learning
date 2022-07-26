#!bin/bash
# These images target a aarch64 machine (e.g. RPi) but you'd probably will be building these images on a x86_64 machine. To achieve this you'll need qemu. 
# You should enable this before building the images by doing:
#  docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
docker-compose up --build
#docker inspect mlflow
