from getpass import getpass
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain
import pickle

# HUGGINGFACEHUB_API_TOKEN = getpass()
import os
def read_list():
    # for reading also binary mode is important
    with open('/data/urlcontent.pickle', 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list
    
# documents=read_list()
# document_string=' '.join(str(v) for v in documents)
    
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'

question = "What is fogify?"

template = """
Some information about the fogify tool: 
Fogify is an emulation Framework easing the modeling, deployment 
and experimentation of fog testbeds. 
Fogify provides a toolset to: model complex 
fog topologies comprised of heterogeneous resources,
network capabilities and QoS criteria; deploy the modelled
configuration and services using popular containerized 
infrastructure-as-code descriptions to a cloud or local
environment; experiment, measure and evaluate the deployment 
by injecting faults and adapting the configuration at runtime 
to test different "what-if" scenarios that reveal the
limitations of a service before introduced to the public.

Question: {question}

Answer: """

prompt = PromptTemplate(input_variables=[ 'question'],template=template)
repo_id = "gpt2"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 1.0, "max_length":500}
)

llm_chain = LLMChain(prompt=prompt, llm=llm,verbose=True)

print(llm_chain.run({'question':question}))

# import pickle
# def read_list():
#     # for reading also binary mode is important
#     with open('/data/urlcontent.pickle', 'rb') as fp:
#         n_list = pickle.load(fp)
#         return n_list
# from transformers import GPT2LMHeadModel, GPT2Tokenizer

# model_name = 'gpt2'  # Replace with the model you want to use
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# model = GPT2LMHeadModel.from_pretrained(model_name)

# documents=read_list()
# trainingDS = tokenizer.encode(documents, return_tensors='pt')

# from transformers import Trainer, TrainingArguments

# # Define your training arguments
# training_args = TrainingArguments(
#     output_dir='./results',  # Directory to save checkpoints and results
#     num_train_epochs=3,      # Number of epochs
#     per_device_train_batch_size=4,
#     save_steps=500,           # Save model checkpoint every X steps
#     logging_steps=100,        # Log metrics every X steps
#     evaluation_strategy="steps",
#     eval_steps=1000,
#     save_total_limit=2,
# )

# # Define Trainer object
# trainer = Trainer(
#     model=model,                         # The model to be trained
#     args=training_args,                  # Training arguments
#     train_dataset=trainingDS, # Your training dataset
#     tokenizer=tokenizer                   # The tokenizer for encoding
# )

# # Start training
# trainer.train()
