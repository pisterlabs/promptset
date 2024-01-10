import os
from dotenv import load_dotenv, find_dotenv
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import YoutubeLoader
import textwrap

# --------------------------------------------------------------
# Load the HuggingFaceHub API token from the .env file
# --------------------------------------------------------------

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]


# --------------------------------------------------------------
# Load the LLM model from the HuggingFaceHub
# --------------------------------------------------------------

repo_id = "tiiuae/falcon-7b-instruct"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
falcon_llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 500}
)


# --------------------------------------------------------------
# Create a PromptTemplate and LLMChain
# --------------------------------------------------------------
template = """What do you think about {question}?"""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=falcon_llm)

# --------------------------------------------------------------
# Create a ConversationChain
# --------------------------------------------------------------
conversation_chain = ConversationChain(llm=falcon_llm, verbose=True, memory=ConversationBufferMemory())
conversation_response_chain = ConversationChain(llm=falcon_llm, verbose=True, memory=ConversationBufferMemory())
# --------------------------------------------------------------
# Run the LLMChain
# --------------------------------------------------------------

question = "What is the meaning of life?"
response = llm_chain.run(question)
wrapped_text = textwrap.fill(
    response, width=100, break_long_words=False, replace_whitespace=False
)
print(wrapped_text)


# --------------------------------------------------------------
# Load a video transcript from YouTube
# --------------------------------------------------------------

video_url = "https://www.youtube.com/watch?v=Jv79l1b-eoI"
loader = YoutubeLoader.from_youtube_url(video_url)
transcript = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000)
docs = text_splitter.split_documents(transcript)

# --------------------------------------------------------------
# Summarization with LangChain
# --------------------------------------------------------------

# Add map_prompt and combine_prompt to the chain for custom summarization
chain = load_summarize_chain(falcon_llm, chain_type="map_reduce", verbose=True)
print(chain.llm_chain.prompt.template)
print(chain.combine_document_chain.llm_chain.prompt.template)

# --------------------------------------------------------------
# Test the Falcon model with text summarization
# --------------------------------------------------------------

output_summary = chain.run(docs)
wrapped_text = textwrap.fill(
    output_summary, width=100, break_long_words=False, replace_whitespace=False
)
print(wrapped_text)

output_summary = output_summary

# Counter to track the number of iterations
counter = 0

while counter < 3:
    # The first prediction step (replace with your prediction function)
    response = conversation_chain.predict(input=output_summary)
    
    # Wrap and print the response text
    wrapped_text = textwrap.fill(response, width=100, break_long_words=False, replace_whitespace=False)
    print(wrapped_text)
    
    # The second prediction step (replace with your prediction function)
    response_chain = conversation_response_chain.predict(input=response)
    
    # Wrap and print the response chain text
    wrapped_text = textwrap.fill(response_chain, width=100, break_long_words=False, replace_whitespace=False)
    print(wrapped_text)
    
    # Set the output of the response chain as the new output summary for the next iteration
    output_summary = response_chain
    
    # Increment the counter
    counter += 1







