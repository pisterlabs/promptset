#!/bin/bash
#
# hello-world.sh
# My CHTC job
#
# print a 'hello' message to the job's terminal output:
echo "Hello CHTC from Job ID $1 running on `whoami`@`hostname`"

# Note: Removed -zv from tar command to save time.

# Setting environment variables
export TF_ENABLE_ONEDNN_OPTS=0
export TRANSFORMERS_CACHE=$PWD/model_cache  # .cache directory where the transformers library downloads model weights
export HUGGINGFACE_API_KEY=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

# # Moving files from staging to working directory
# if [ -f /staging/djpaul2/model_cache.tar.gz ]; then
#     # If the model_cache.tar.gz file exists, extract it
#     tar -xf /staging/djpaul2/model_cache.tar.gz -C ./
# fi

# Run Script and move output to staging
time python3 main.py
time python3 save.py $1

# Clean up
# if [ ! -f /staging/djpaul2/model_cache.tar.gz ]; then
#     # If the model_cache.tar.gz file doesn't exists, create and send it
#     tar -cf model_cache.tar.gz ./model_cache
#     mv model_cache.tar.gz /staging/djpaul2/
# fi
rm -rf model_cache
