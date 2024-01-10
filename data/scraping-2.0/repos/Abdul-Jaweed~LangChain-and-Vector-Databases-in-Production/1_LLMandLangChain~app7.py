## Summarization chain example:

# LangChain prompts can be found in various use cases, such as summarization or question-answering chains. For example, when creating a summarization chain, LangChain enables interaction with an external data source to fetch data for use in the generation step. This could involve summarizing a lengthy piece of text or answering questions using specific data sources.

# The following code will initialize the language model using OpenAI class with a temperature of 0 - because we want deterministic output.  The load_summarize_chain function accepts an instance of the language model and returns a pre-built summarization chain. Lastly, the PyPDFLoader class is responsible for loading PDF files and converting them into a format suitable for processing by LangChain. 

# It is important to note that you need to install the pypdf package to run the following code. Although it is highly recommended to install the latest versions of this package, the codes have been tested on version 3.10.0. Please refer to course introduction lesson for more information on installing packages. 



from langchain import OpenAI, PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader

import os
from dotenv import load_dotenv

load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")

# Initialize language model

llm = OpenAI(
    openai_api_key=apikey,
    model="text-davinci-003",
    temperature=0
)

# load the summarization chain

summarize_chain  = load_summarize_chain(llm)

# load the document using PyPDFLoader

document_loader = PyPDFLoader(file_path="file-path")

document = document_loader.load()

# Summarize the document

summary = summarize_chain(document)

print(summary['output_text'])

