import os
from dotenv import load_dotenv, find_dotenv
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain, OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader
import textwrap




# --------------------------------------------------------------
# Create a PromptTemplate and LLMChain
# --------------------------------------------------------------
template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=prashant_llm)


# --------------------------------------------------------------
# Run the LLMChain
# --------------------------------------------------------------

question = "How do I make a sandwich?"
response = llm_chain.run(question)
wrapped_text = textwrap.fill(
    response, width=100, break_long_words=False, replace_whitespace=False
)
print(wrapped_text)



# Add map_prompt and combine_prompt to the chain for custom summarization
chain = load_summarize_chain(prashant_llm, chain_type="map_reduce", verbose=True)
print(chain.llm_chain.prompt.template)
print(chain.combine_document_chain.llm_chain.prompt.template)

# --------------------------------------------------------------
# Test the prashant model with text summarization
# --------------------------------------------------------------

output_summary = chain.run(docs)
wrapped_text = textwrap.fill(
    output_summary, width=100, break_long_words=False, replace_whitespace=False
)
print(wrapped_text)


# --------------------------------------------------------------
# Load an OpenAI model for comparison
# --------------------------------------------------------------

openai_llm = OpenAI(
    model_name="text-davinci-003", temperature=0.1, max_tokens=500
)  # max token length is 4097
chain = load_summarize_chain(openai_llm, chain_type="map_reduce", verbose=True)
output_summary = chain.run(docs)
wrapped_text = textwrap.fill(
    output_summary, width=100, break_long_words=False, replace_whitespace=False
)
print(wrapped_text)
