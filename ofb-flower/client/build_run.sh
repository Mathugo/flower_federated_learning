# These images target a aarch64 machine (e.g. RPi) but you'd probably will be building these images on a x86_64 machine. To achieve this you'll need qemu. 
# You should enable this before building the images by doing:
#  docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
sudo docker build -t mathugo/ofb-flower .
sudo docker run -it --rm mathugo/ofb-flower --name usine_two --server_address localhost --server_port 4040 --cid 3 --model hugonet --n_classes=3 --data_augmentation --mlflow_server_ip localhost --mlflow_server_port 4040