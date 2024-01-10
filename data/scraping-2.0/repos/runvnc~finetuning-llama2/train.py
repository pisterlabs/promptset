import sagemaker
import time
import json

from datasets import Dataset
from langchain.document_loaders import WebBaseLoader
from sagemaker.huggingface import HuggingFace 
from huggingface_hub import HfFolder

import init_sagemaker


def fine_tune(s3_path,
              pretrained_model_id="meta-llama/Llama-2-13b-chat-hf",
              batch_size = 1, learning_rate = 2e-4,
              epochs = 20,
              instance_type='ml.g5.12xlarge'):

    sess, _, role = init_sagemaker.init_session()

    print("pretrained model id = ", pretrained_model_id)

    job_name = f'huggingface-qlora-{pretrained_model_id.replace("/", "-")}-{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}'

    hyperparameters ={
      'model_id': pretrained_model_id,                  # pre-trained model
      'dataset_path': '/opt/ml/input/data/training',    # path where sagemaker will save training dataset
      'epochs': epochs,                                 # number of training epochs
      'per_device_train_batch_size': batch_size,        # batch size for training
      'lr': learning_rate,                              # learning rate used during training
      'hf_token': HfFolder.get_token(),                 # huggingface token to access llama 2
      'merge_weights': False,                           # wether to merge LoRA into the model (needs more memory)
      'nproc_per_node': 4
      }
    # nproc_per_node=4 ? (4 GPU?)

    huggingface_estimator = HuggingFace(
        entry_point          = 'run_clm.py',      # train script
        source_dir           = 'scripts',         # directory which includes all the files needed for training
        instance_type        = instance_type,     # instances type used for the training job
        instance_count       = 1,                 # the number of instances used for training
        base_job_name        = job_name,          # the name of the training job
        role                 = role,              # Iam role used in training job to access AWS ressources, e.g. S3
        volume_size          = 300,               # the size of the EBS volume in GB
        transformers_version = '4.28',            # the transformers version used in the training job
        pytorch_version      = '2.0',             # the pytorch_version version used in the training job
        py_version           = 'py310',           # the python version used in the training job
        hyperparameters      =  hyperparameters,  # the hyperparameters passed to the training job
        environment          = { "HUGGINGFACE_HUB_CACHE": "/tmp/.cache", "CUDA_VISIBLE_DEVICES": "0,1,2,3" }, # set env variable to cache models in /tmp
    )

    # define a data input dictonary with our uploaded s3 uris
    full_s3_path = 's3://' + sess.default_bucket() + '/' + s3_path
 
    data = {'training': full_s3_path}

    # starting the train job with our uploaded datasets as input
    huggingface_estimator.fit(data, wait=False)

    from sagemaker.huggingface import get_huggingface_llm_image_uri

    # retrieve the llm image uri
    llm_image = get_huggingface_llm_image_uri(
      "huggingface",
      version="0.9.3"
    )

    print(f"llm image uri: {llm_image}")

    return llm_image

