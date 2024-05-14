#!/bin/bash
#
# hello-world.sh
# My CHTC job
#
# print a 'hello' message to the job's terminal output:
echo "Hello CHTC from Job $1 running on `whoami`@`hostname`"

export TF_ENABLE_ONEDNN_OPTS=0

# extract model weights (removing -v flag to keep output clean)
# tar -xzf /staging/djpaul2/gemma_model_saved.tar.gz -C ./

pip3 install --upgrade pip
pip3 install -r requirements.txt  # TODO: Use venv later

# Run Script and move output to staging
python3 run.py > out.txt
python3 run.py
mv out.txt /staging/djpaul2/out.txt

