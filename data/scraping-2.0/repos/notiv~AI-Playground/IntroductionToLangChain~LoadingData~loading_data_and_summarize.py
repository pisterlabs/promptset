#%%
import os

import hvac
from langchain.chains import ConversationChain, LLMChain, SimpleSequentialChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               PromptTemplate, SystemMessagePromptTemplate)
from langchain.schema import BaseOutputParser
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter
)

# Retrieve the openai_test_token from Vault
client = hvac.Client(url='http://localhost:8200')
client.token = os.environ['VAULT_TOKEN']

#%%
if not client.is_authenticated():
    print('Client is not authenticated.')
    exit(1)

try:
    openai_token = client.secrets.kv.v2.read_secret(path='openai', mount_point='kv')['data']['data']['openai_test_token']
     #.v2.read_secret(path='secrets/api_readonly/openai_test_token')
except hvac.exceptions.InvalidPath:
    print('The secret path is invalid.')
    exit(1)

# %%
file_path = 'ESLII_print12_toc.pdf'
sl_loader = PyPDFLoader(file_path=file_path)
sl_data = sl_loader.load_and_split()

sl_data[0]

# %%
# split on "\n\n"
splitter1 = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
)

# split ["\n\n", "\n", " ", ""]
splitter2 = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
)

sl_data1 = sl_loader.load_and_split(text_splitter=splitter1)
sl_data2 = sl_loader.load_and_split(text_splitter=splitter2)
# %%
folder_path = '.'

mixed_loader = DirectoryLoader(path=folder_path,
                               use_multithreading=True,
                               show_progress=True)

mixed_data = mixed_loader.load_and_split()
# %%
llm = ChatOpenAI(openai_api_key=openai_token)
chain = load_summarize_chain(
    llm=llm,
    chain_type='stuff',
)
chain.run(sl_data[:2])

# %%
template = """
Write a concise summary of the following in german:
"{text}"

CONCISE SUMMARY IN GERMAN:
"""

# %%
prompt = PromptTemplate.from_template(template)

chain = load_summarize_chain(
    llm=llm,
    prompt=prompt
)

chain.run(sl_data[:2])
# %%
chain = load_summarize_chain(
    llm=llm,
    chain_type='map_reduce',
)   

chain.run(sl_data[:20])
# %%
map_template = """The following is a set of documents
{text}
Based on this list of docs, please identify the main themes.
Helpful Answer:
"""

combine_template = """The following is a set of summaries:

{text}

Take these and distill it into a final, consolidated list of the main themes.
Return that list as a comma separated list.
Helpful Answer:
"""

# %%
map_prompt = PromptTemplate.from_template(map_template)
combine_prompt = PromptTemplate.from_template(combine_template)

chain = load_summarize_chain(
    llm=llm,
    map_prompt=map_prompt,
    combine_prompt=combine_prompt,
    chain_type='map_reduce',
    verbose=True
)
# %%
chain.run(sl_data[:40])
# %%
chain = load_summarize_chain(
    llm=llm,
    chain_type='refine',
    verbose=True
)

chain.run(sl_data[:20])
# %%
initial_template = """
Extract the most relevant themes from the following:
{text}

THEMES:"""

refine_template = """
Your job is to extract the most relevant themes.
We have provided an existing list of themes up to a certain point:
{existing_answer}
We have the opportunity to refine the existing list (only if needed) with some
more context below.
--------
{text}
--------
Given the new context, refine the original list. If the context isn't useful, 
return the original list. Return that list as a comma separated list.

LIST:
"""

initial_prompt = PromptTemplate.from_template(initial_template)
refine_prompt = PromptTemplate.from_template(refine_template)

chain = load_summarize_chain(
    llm=llm,
    chain_type='refine',
    question_prompt=initial_prompt,
    refine_prompt=refine_prompt,
    verbose=True
)

chain.run(sl_data[:20])
# %%
