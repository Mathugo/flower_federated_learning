# These images target a aarch64 machine (e.g. RPi) but you'd probably will be building these images on a x86_64 machine. To achieve this you'll need qemu. 
# You should enable this before building the images by doing:
#  docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
sudo docker build -t mathugo/ofb-flower .
sudo docker run -it --rm mathugo/ofb-flower --server=localhost:4444 --cid=1 --model=Net