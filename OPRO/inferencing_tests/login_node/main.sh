#!/bin/bash
#
# hello-world.sh
# My CHTC job
#
# print a 'hello' message to the job's terminal output:
echo "Hello CHTC from Job $1 running on `whoami`@`hostname`"

# extract model weights (removing -v flag to keep output clean)
# tar -xzf /staging/djpaul2/gemma_model_saved.tar.gz -C ./
# tar -xzf /staging/djpaul2/venv.tar.gz -C ./
tar -xzvf ../gemma_model_saved.tar.gz -C ./
tar -xzvf ../venv.tar.gz -C ./

# activate virtual environment
source venv/bin/activate
python gemma.py 

# deactivate virtual environment
deactivate

# clean up
rm -rf venv
rm -rf gemma_model_saved
