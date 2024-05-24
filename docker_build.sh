#!/bin/bash

# Include the configuration file
source config.sh

# Build the image from the directory where the Dockerfile is saved
docker build -t $IMAGE_NAME .
