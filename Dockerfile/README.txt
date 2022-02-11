# Build docker image
docker build -t IMAGE_NAME .

# Run docker container
#docker run --runtime=nvidia -v /media/share/:/media/share -it --net=host --name=CONTAINER_NAME IMAGE_NAME:latest bash
docker run --gpus all -v /data/vagrant/:/media/share -v /home/vagrant/.Xauthority:/root/.Xauthority:rw -e DISPLAY=$DISPLAY -it --net=host --name=peiyan_tensorflow_1 peiyan_tensorflow:latest bash

docker run --gpus all -v /media/share/:/media/share -it --net=host --name=CONTAINER_NAME IMAGE_NAME:latest bash
