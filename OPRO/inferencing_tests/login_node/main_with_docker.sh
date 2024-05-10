#!/bin/bash
#
# hello-world.sh
# My CHTC job
#
# print a 'hello' message to the job's terminal output:
echo "Hello CHTC from Job $1 running on `whoami`@`hostname`"

# extract model weights (removing -v flag to keep output clean)
tar -xzf /staging/djpaul2/gemma_model_saved.tar.gz -C ./

# Run the model
python gemma.py
