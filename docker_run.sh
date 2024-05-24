#!/bin/bash

# Include the configuration file
source config.sh

# Run Docker container
docker run -d --gpus '"device=2,3,4,5"' --mount type=bind,src=$MOUNT_SRC,target=$MOUNT_TARGET $IMAGE_NAME
