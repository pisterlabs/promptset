#!/usr/bin/env python
# coding: utf-8

# # Notebook Setup

# In[2]:


get_ipython().system('pip install "transformers==4.31.0" "datasets[s3]==2.13.0" sagemaker --upgrade --quiet')


# In[3]:


get_ipython().system('pip install huggingface_hub')


# In[4]:


get_ipython().system('pip install datasets')


# In[5]:


get_ipython().system('pip install --upgrade langchain==0.0.249')


# In[6]:


get_ipython().system('pip install transformers')


# In[7]:


get_ipython().system('pip install bs4')


# In[8]:


get_ipython().system('pip install --upgrade accelerate')


# In[ ]:


get_ipython().system('huggingface-cli login --token ***enter huggingface token***')


# In[9]:


import sagemaker
import boto3
import time
import json

from datasets import Dataset
from langchain.document_loaders import WebBaseLoader
from random import randint
from itertools import chain
from functools import partial
from transformers import AutoTokenizer
from sagemaker.huggingface import HuggingFace, HuggingFaceModel
from huggingface_hub import HfFolder


# In[10]:


sess = sagemaker.Session()
# sagemaker session bucket -> used for uploading data, models and logs
# sagemaker will automatically create this bucket if it not exists
sagemaker_session_bucket=None
if sagemaker_session_bucket is None and sess is not None:
    # set to default bucket if a bucket name is not given
    sagemaker_session_bucket = sess.default_bucket()

try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']

sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

print(f"sagemaker role arn: {role}")
print(f"sagemaker bucket: {sess.default_bucket()}")
print(f"sagemaker session region: {sess.boto_region_name}")


# # Data retrieval

# In[12]:


loader = WebBaseLoader(["https://aws.amazon.com/blogs/aws/preview-enable-foundation-models-to-complete-tasks-with-agents-for-amazon-bedrock/", "https://aws.amazon.com/blogs/aws/aws-entity-resolution-match-and-link-related-records-from-multiple-applications-and-data-stores/", "https://aws.amazon.com/blogs/database/the-role-of-vector-datastores-in-generative-ai-applications/", "https://aws.amazon.com/blogs/big-data/introducing-the-vector-engine-for-amazon-opensearch-serverless-now-in-preview/", "https://aws.amazon.com/blogs/big-data/build-data-integration-jobs-with-ai-companion-on-aws-glue-studio-notebook-powered-by-amazon-codewhisperer/", "https://aws.amazon.com/blogs/aws/new-amazon-ec2-p5-instances-powered-by-nvidia-h100-tensor-core-gpus-for-accelerating-generative-ai-and-hpc-applications/"])


# In[13]:


data = loader.load()
data


# # Data processing

# In[14]:


def strip_spaces(doc):
    return {"text": doc.page_content.replace("  ", "")}


# In[15]:


stripped_data = list(map(strip_spaces, data))
stripped_data


# In[16]:


dataset = Dataset.from_list(stripped_data)
dataset


# In[17]:


model_id = "meta-llama/Llama-2-13b-hf" # sharded weights
tokenizer = AutoTokenizer.from_pretrained(model_id,use_auth_token=True)
tokenizer.pad_token = tokenizer.eos_token


# In[18]:


# empty list to save remainder from batches to use in next batch
remainder = {"input_ids": [], "attention_mask": [], "token_type_ids": []}

def chunk(sample, chunk_length=2048):
    # define global remainder variable to save remainder from batches to use in next batch
    global remainder
    # Concatenate all texts and add remainder from previous batch
    concatenated_examples = {k: list(chain(*sample[k])) for k in sample.keys()}
    concatenated_examples = {k: remainder[k] + concatenated_examples[k] for k in concatenated_examples.keys()}
    # get total number of tokens for batch
    batch_total_length = len(concatenated_examples[list(sample.keys())[0]])

    # get max number of chunks for batch
    if batch_total_length >= chunk_length:
        batch_chunk_length = (batch_total_length // chunk_length) * chunk_length

    # Split by chunks of max_len.
    result = {
        k: [t[i : i + chunk_length] for i in range(0, batch_chunk_length, chunk_length)]
        for k, t in concatenated_examples.items()
    }
    # add remainder to global variable for next batch
    remainder = {k: concatenated_examples[k][batch_chunk_length:] for k in concatenated_examples.keys()}
    # prepare labels
    result["labels"] = result["input_ids"].copy()
    return result


# tokenize and chunk dataset
lm_dataset = dataset.map(
    lambda sample: tokenizer(sample["text"]), batched=True, remove_columns=list(dataset.features)
).map(
    partial(chunk, chunk_length=4096),
    batched=True,
)


# Print total number of samples
print(f"Total number of training samples: {len(lm_dataset)}")


# In[19]:


def sum_dataset_arrays(dataset):
    total = 0
    for i in range(0,8):
        total = total + len(lm_dataset[i]['input_ids'])
    return total
print(f"Total number of training tokens: {sum_dataset_arrays(lm_dataset)}")


# In[ ]:


# save train_dataset to s3
training_input_path = f's3://{sess.default_bucket()}/processed/llama/genai-nyc-summit/train'
lm_dataset.save_to_disk(training_input_path)

print("uploaded data to:")
print(f"training dataset to: {training_input_path}")


# # Fine-tuning

# In[ ]:


# define Training Job Name
job_name = f'huggingface-qlora-{model_id.replace("/", "-")}-{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}'

# hyperparameters, which are passed into the training job
hyperparameters ={
  'model_id': model_id,                             # pre-trained model
  'dataset_path': '/opt/ml/input/data/training',    # path where sagemaker will save training dataset
  'epochs': 20,                                      # number of training epochs
  'per_device_train_batch_size': 2,                 # batch size for training
  'lr': 2e-4,                                       # learning rate used during training
  'hf_token': HfFolder.get_token(),                 # huggingface token to access llama 2
  'merge_weights': True,                            # wether to merge LoRA into the model (needs more memory)
}

# create the Estimator
huggingface_estimator = HuggingFace(
    entry_point          = 'run_clm.py',      # train script
    source_dir           = 'scripts',         # directory which includes all the files needed for training
    instance_type        = 'ml.g5.4xlarge',   # instances type used for the training job
    instance_count       = 1,                 # the number of instances used for training
    base_job_name        = job_name,          # the name of the training job
    role                 = role,              # Iam role used in training job to access AWS ressources, e.g. S3
    volume_size          = 300,               # the size of the EBS volume in GB
    transformers_version = '4.28',            # the transformers version used in the training job
    pytorch_version      = '2.0',             # the pytorch_version version used in the training job
    py_version           = 'py310',           # the python version used in the training job
    hyperparameters      =  hyperparameters,  # the hyperparameters passed to the training job
    environment          = { "HUGGINGFACE_HUB_CACHE": "/tmp/.cache" }, # set env variable to cache models in /tmp
)


# In[ ]:


# define a data input dictonary with our uploaded s3 uris
data = {'training': training_input_path}

# starting the train job with our uploaded datasets as input
huggingface_estimator.fit(data, wait=False)


# # Deployment

# In[ ]:


from sagemaker.huggingface import get_huggingface_llm_image_uri

# retrieve the llm image uri
llm_image = get_huggingface_llm_image_uri(
  "huggingface",
  version="0.8.2"
)

# print ecr image uri
print(f"llm image uri: {llm_image}")


# In[ ]:


# sagemaker config
instance_type = "ml.g5.12xlarge"
number_of_gpu = 4
health_check_timeout = 300

# TGI config
config = {
  'HF_MODEL_ID': "/opt/ml/model", # path to where sagemaker stores the model
  'SM_NUM_GPUS': json.dumps(number_of_gpu), # Number of GPU used per replica
  'MAX_INPUT_LENGTH': json.dumps(1024),  # Max length of input text
  'MAX_TOTAL_TOKENS': json.dumps(2048),  # Max length of the generation (including input text),
  # 'HF_MODEL_QUANTIZE': "bitsandbytes", # comment in to quantize
}

# create HuggingFaceModel
llm_model = HuggingFaceModel(
  role=role,
  image_uri=llm_image,
  #model_data="s3://sagemaker-us-east-1-308819823671/huggingface-qlora-llama2-13b-chat-2023--2023-08-02-08-54-16-604/output/model.tar.gz",
  model_data="s3://sagemaker-us-east-1-308819823671/huggingface-qlora-meta-llama-Llama-2-13-2023-09-01-16-39-25-384/output/model.tar.gz",
  env=config
)


# In[ ]:


# Deploy model to an endpoint
# https://sagemaker.readthedocs.io/en/stable/api/inference/model.html#sagemaker.model.Model.deploy
llm = llm_model.deploy(
  #endpoint_name="llama-2-13b-chat-hf-nyc-finetuned", # alternatively "llama-2-13b-hf-nyc-finetuned" 
  endpoint_name="llama-2-13b-hf-nyc-finetuned", # alternatively "llama-2-13b-hf-nyc-finetuned"  
  initial_instance_count=1,
  instance_type=instance_type,
  # volume_size=400, # If using an instance with local SSD storage, volume_size must be None, e.g. p4 but not p3
  container_startup_health_check_timeout=health_check_timeout, # 10 minutes to be able to load the model
)


# # Inference

# In[33]:


basic_endpoint = 'jumpstart-dft-meta-textgeneration-llama-2-13b' 
basic_endpoint_ft = 'llama-2-13b-hf-nyc-finetuned'
chat_endpoint = 'jumpstart-dft-meta-textgeneration-llama-2-13b-f'
chat_endpoint_ft = 'llama-2-13b-chat-hf-nyc-finetuned'


# In[34]:


def query_endpoint(payload, endpoint_name):
    client = boto3.client("sagemaker-runtime")
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
        CustomAttributes="accept_eula=true",
    )
    response = response["Body"].read().decode("utf8")
    response = json.loads(response)
    return response


# # Base vs. chat model
# ## LLaMA2 

# In[35]:


prompt = "Hi! I'm Aris and I am wondering what I should do today in sunny Athens."
print(f'{basic_endpoint}: {query_endpoint({"inputs": prompt, "parameters": {"max_new_tokens": 200, "top_p": 0.9, "temperature": 0.01, "return_full_text": False}}, basic_endpoint)}')


# ## LLaMA2 finetuned on NYC summit blogs

# In[36]:


prompt = "Hi! I'm Aris and I am wondering what I should do today in sunny Athens."
print(f'{basic_endpoint_ft}: {query_endpoint({"inputs": prompt, "parameters": {"max_new_tokens": 200, "top_p": 0.9, "temperature": 0.01, "return_full_text": False}}, basic_endpoint_ft)}')


# ## LLaMA2-chat

# In[38]:


prompt = "Hi! I'm Aris and I am wondering what I should do today in sunny Athens."
print(f'{chat_endpoint}: {query_endpoint({"inputs": [[{"role": "user", "content": prompt}]], "parameters": {"max_new_tokens": 200, "top_p": 0.9, "temperature": 0.01}}, chat_endpoint)}')


# ## LLaMA2-chat finetuned on NYC summit blogs

# In[39]:


prompt = "Hi! I'm Aris and I am wondering what I should do today in sunny Athens."
print(f'{chat_endpoint_ft}: {query_endpoint({"inputs": json.dumps([[{"role": "user", "content": prompt}]]), "parameters": {"max_new_tokens": 200, "top_p": 0.9, "temperature": 0.01}}, chat_endpoint_ft)}')


# In[40]:


prompt = "<s> [INST] Hi! I'm Aris and I am wondering what I should do today in sunny Athens. [/INST]"
print(f'{chat_endpoint_ft}: {query_endpoint({"inputs": prompt, "parameters": {"max_new_tokens": 200, "top_p": 0.9, "temperature": 0.01}}, chat_endpoint_ft)}')


# # Do the different models know what P5 instances are?
# ## LLaMA2-13b 

# In[27]:


prompt = "Amazon EC2 P5 instances are equipped with GPUs of the type"
print(f'{basic_endpoint}: {query_endpoint({"inputs": prompt, "parameters": {"max_new_tokens": 200, "top_p": 0.9, "temperature": 0.01, "return_full_text": False}}, basic_endpoint)}')


# ## LLaMA2-13b finetuned on NYC summit blogs

# In[29]:


prompt = "Amazon EC2 P5 instances are equipped with GPUs of the type"
print(f'{basic_endpoint_ft}: {query_endpoint({"inputs": prompt, "parameters": {"max_new_tokens": 200, "top_p": 0.9, "temperature": 0.01, "return_full_text": False}}, basic_endpoint_ft)}')


# ## LLaMA2-13b-chat

# In[30]:


prompt = "What are Amazon EC2 P5 instances? Which kind of GPUs are they equipped with?"
print(f'{chat_endpoint}: {query_endpoint({"inputs": [[{"role": "user", "content": prompt}]], "parameters": {"max_new_tokens": 200, "top_p": 0.9, "temperature": 0.01}}, chat_endpoint)}')


# ## LLaMA2-13b-chat finetuned on NYC summit blogs

# In[31]:


prompt = "What are Amazon EC2 P5 instances? Which kind of GPUs are they equipped with?"
print(f'{chat_endpoint_ft}: {query_endpoint({"inputs": json.dumps([[{"role": "user", "content": prompt}]]), "parameters": {"max_new_tokens": 200, "top_p": 0.9, "temperature": 0.01, "return_full_text": False}}, chat_endpoint_ft)}')


# In[41]:


prompt = "<s> [INST] What are Amazon EC2 P5 instances? Which kind of GPUs are they equipped with? [/INST]"
print(f'{chat_endpoint_ft}: {query_endpoint({"inputs": prompt, "parameters": {"max_new_tokens": 200, "top_p": 0.9, "temperature": 0.01, "return_full_text": False}}, chat_endpoint_ft)}')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


prompt = """
Blogpost conclusion: 
In conclusion, this blog post delves into the critical process of infusing domain-specific knowledge into large language models (LLMs) like LLaMA2, emphasizing the importance of addressing challenges related to helpfulness, honesty, and harmlessness when designing LLM-powered applications for enterprise-grade quality. The primary focus here is on the parametric approach to fine-tuning, which efficiently injects niche expertise into foundation models without compromising their general linguistic capabilities.The blog highlights the steps involved in fine-tuning LLaMA2 using parameter-efficient fine-tuning techniques, such as the qLoRA approach, and how this process can be conducted on Amazon SageMaker. By adopting this approach, practitioners can adapt LLaMA2 to specific domains, ensuring that the models remain up-to-date with recent knowledge even beyond their original training data. The article also underscores the versatility of this approach, showing that it can be applied to models like LLaMA2-chat, which have already undergone task-specific fine-tuning. This opens up opportunities to infuse knowledge into LLMs without the need for extensive instruction or chat-based fine-tuning, preserving their task-specific nature.
Task: 
Please extract the main takeaways from this blogpost.
"""

print(f'{chat_endpoint}: {query_endpoint({"inputs": [[{"role": "user", "content": prompt}]], "parameters": {"max_new_tokens": 200, "top_p": 0.9, "temperature": 0.01}}, chat_endpoint)}')


# In[ ]:




