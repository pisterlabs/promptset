FROM ollama/ollama:latest as ollama

# Deriving the latest base image
FROM python:latest

# Labels as key value pair
LABEL maintainer="djp4u1"

# ADD can also copy compressed (tarball) files and automatically extract the content at the destination. 
# This feature only applies to locally stored compressed files and directories.
# The command extracts a compressed source only if it is in a recognized compression format 
# (based on the file's contents, not the extension). The recognized compression formats include identity, gzip, bzip, and xz.
# ADD gemma_model_saved.tar.gz ./
# COPY .env ./
# COPY gemma.py ./
COPY requirements.txt ./

RUN pip install -r requirements.txt

# CMD [ "python", "./gemma.py"]
