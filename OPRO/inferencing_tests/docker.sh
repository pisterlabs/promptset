IMAGE_NAME="djp4u1/gemma_img:latest"

# Docker Cheatsheet: https://docs.docker.com/get-started/docker_cheatsheet.pdf
docker build -t $IMAGE_NAME . 
docker run --name my_container --gpus=all $IMAGE_NAME
docker rm my_container # docker container prune also works
docker rmi $IMAGE_NAME # docker image prune also works


# NOTE: CHTC uses something similar to the following to run jobs in Docker containers:

# docker run --user $(id -u):$(id -g) --rm=true -it \
#   -v $(pwd):/scratch -w /scratch \
#   username/imagename:tag /bin/bash

# What Do All the Options Mean?
# -it:                  interactive flag
# --rm=true:            after we exit, this will clean up the runnining container so Docker uses less disk space.
# username/image:tag:   which container to start
# /bin/bash:            tells Docker that when the container starts, we want a command line (bash) inside to run commands

# The options that we have added for this example are used in CHTC to make jobs run successfully and securely.
# --user $(id -u):$(id -g):     runs the container with more restrictive permissions
# -v $(pwd):/scratch:           Put the current working directory (pwd) into the container but call it /scratch. 
#                               In CHTC, this working directory will be the job’s usual working directory.
# -w /scratch:                  when the container starts, make /scratch the working directory
