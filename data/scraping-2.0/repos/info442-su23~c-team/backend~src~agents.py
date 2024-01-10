import asyncio
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv
# main imports
#from prompts import label_chat_prompt, table_contents_chat_prompt, summary_chat_prompt
# test imports
from src.prompts import label_chat_prompt, table_contents_chat_prompt, summary_chat_prompt
# Load environment variables from the .env file
load_dotenv('../.env')

# Initialize the GPT-4 model with a temperature of 0
gpt_4_0 = ChatOpenAI(model_name="gpt-4", temperature=0)

# Function to create a new LLMChain with the given prompt
def create_chain(prompt):
    return LLMChain(
        llm=gpt_4_0,  # Use the GPT-4 model
        prompt=prompt(),  # Use the given prompt
        memory=ConversationBufferMemory(),  # Use a ConversationBufferMemory for memory
    )

# Function to process a list of text chunks with a given chain
async def process_text(chain, text_chunks):
    # Run the chain for each text chunk
    for chunk in text_chunks:
        await chain.arun(chunk)
    # Extract the content of AI messages from the chain's memory
    messages = [message.content for message in chain.memory.chat_memory.messages if message.__class__.__name__ == 'AIMessage']
    return messages

# Function to process a list of labels with a given chain
async def process_labels(chain, labels):
    labels_string = '[' + ', '.join(labels) + ']'
    await chain.arun(labels_string)
    messages = [message.content for message in chain.memory.chat_memory.messages if message.__class__.__name__ == 'AIMessage']
    return messages


# Create chains for labeling, summarizing, and creating a table of contents
label_chain = create_chain(label_chat_prompt)
summary_chain = create_chain(summary_chat_prompt)
contents_table_chain = create_chain(table_contents_chat_prompt)

# Functions to label, summarize, and create a table of contents for a list of text chunks
async def label(text_chunks): return await process_text(label_chain, text_chunks)
async def summarize(text_chunks): return await process_text(summary_chain, text_chunks)
async def table_of_contents(labels): return await process_labels(contents_table_chain, labels)