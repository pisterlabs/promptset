#!/bin/bash
#
# hello-world.sh
# My CHTC job
#
# print a 'hello' message to the job's terminal output:
echo "Hello CHTC from Job $1 running on `whoami`@`hostname`"

# Setting environment variables
export TF_ENABLE_ONEDNN_OPTS=0
export TRANSFORMERS_CACHE=$PWD/model_cache  # .cache directory where the transformers library downloads model weights

# Moving files from staging to working directory
if [ -f /staging/djpaul2/model_cache.tar.gz ]; then
    # If the model_cache.tar.gz file exists, extract it
    cp /staging/djpaul2/model_cache.tar.gz .
    tar -xzf model_cache.tar.gz
fi

# Run Script and move output to staging
python3 run.py > out.txt
python3 run.py
mv out.txt /staging/djpaul2/

# Clean up
if [ ! -f /staging/djpaul2/model_cache.tar.gz ]; then
    # If the model_cache.tar.gz file doesn't exists, create and send it
    tar -czf model_cache.tar.gz ./model_cache
    mv model_cache.tar.gz /staging/djpaul2/
fi
rm -rf model_cache
