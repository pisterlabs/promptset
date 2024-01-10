#!/usr/bin/python
# -*- coding: utf-8 -*-
#

"""
[env]
# Conda Environment
conda create --name langchain_ai python=3.9.13
conda info --envs
source activate langchain_ai
conda deactivate

# if needed to remove
conda env remove -n [NAME_OF_THE_CONDA_ENVIRONMENT]
conda env remove -n langchain_ai

# update conda 
conda update -n base -c defaults conda

# to export requirements
pip freeze > requirements.txt

# to install
pip install -r requirements.txt

conda install -c conda-forge langchain


# [path]
cd /Users/brunoflaven/Documents/03_git/ia_usages/ia_using_other_llm/


# LAUNCH the file
python samwit_basic_chain.py


# EXPLICATIONS
https://github.com/samwit/langchain-tutorials/blob/382e8db4dc5e01fc400bee8d4146cb1a2e9c3150/ollama/basic.py

# FOR OLLAMA

# To run and chat with Llama 2
ollama run llama2
ollama run llama2-uncensored
ollama run orca-mini


# remove a model
ollama rm llama2
ollama rm orca-mini
ollama rm mistral
ollama rm falcon:7b
ollama rm mistral:text
ollama rm llama2:latest

# list the model
ollama list

# when you are in the model you can use
>>> /?
>>> /list
>>> /set verbose

# to get out from the model
/exit


"""



from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler    

"""
llm = Ollama(model="llama2", 
            #  callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]),
            temperature=0.9,
             )
"""
llm = Ollama(model="orca-mini:latest", 
            #  callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]),
            temperature=0.9,
             )


from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Give me 5 interesting facts about {topic}?",
)

from langchain.chains import LLMChain
chain = LLMChain(llm=llm, 
                 prompt=prompt,
                 verbose=False)

# Run the chain only specifying the input variable.
# print(chain.run("the moon"))
# print(chain.run("Argentina"))
# print(chain.run("Ludwig Wittgenstein"))

print(chain.run("Barack Obama"))
