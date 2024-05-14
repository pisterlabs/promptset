# ABANDONED: This file is abandoned because ollama requires root access to run. :(

# Deriving the latest base image
FROM python:latest

# ADD can also copy compressed (tarball) files and automatically extract the content at the destination. 
# This feature only applies to locally stored compressed files and directories.
# The command extracts a compressed source only if it is in a recognized compression format 
# (based on the file's contents, not the extension). The recognized compression formats include identity, gzip, bzip, and xz.
# ADD gemma_model_saved.tar.gz ./
# COPY .env ./
# COPY gemma.py ./

# WORKDIR is set during runtime, and write permisions are limited. So setting this during runtime to $PWD/.ollama
# ENV OLLAMA_MODELS=/home/.ollama

# To install lshw, we need to update the package list
RUN apt-get update -y

# To help ollama detect gpu, we need to install lshw
RUN apt-get install lshw -y

# Downloading and Installing ollama
RUN curl -fsSL https://ollama.com/install.sh | sh


# TO KEEP THE CONTAINER LIGHTWEIGHT, WE WILL NOT COPY THE MODELS INTO THE CONTAINER
# INSTEAD, WE WILL COPY THEM FROM STAGING.
# COPY ./.ollama /home/.ollama
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# CMD [ "ollama", "serve"]
