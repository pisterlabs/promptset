
import os
import openai
openai.api_key = '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'


from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

LLM = OpenAI()                                                                                                                                  # Initialize an instance of the OpenAI API.

PROMPT_TEMPLATE = """Write a concise summary in circa 200 words of the following:

{text}

CONCISE SUMMARY IN DANISH:"""                                                                                                                   # Define a template for the prompt to be used in generating summaries, which includes a placeholder for the input text and a Danish summary label.

PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["text"])                                                                     # Create a prompt template object with the defined template and input variables, in this case only "text".

def get_summary(pages):
    try:                                                                                                                                        # Try to execute the following code block.
        docs = [Document(page_content=t) for t in pages]                                                                                        # Create a list of Document objects, each containing a page of text from the input pages.
        chain = load_summarize_chain(LLM, chain_type = "map_reduce", return_intermediate_steps=False, map_prompt=PROMPT, combine_prompt=PROMPT) # Load a summarization chain with the specified parameters, using the initialized OpenAI instance and the defined prompt template.
        result = chain({"input_documents": docs}, return_only_outputs=True)                                                                     # Execute the summarization chain with the input documents, and retrieve the output text.
        return result["output_text"]   
                                                                                                             # Return the generated summary text.
    except openai.error.InvalidRequestError:                                                                                                    # If an InvalidRequestError occurs, execute the following code block.
        chunked_pages = []                                                                                                                      # Create an empty list to store chunked pages.
        summarized_pages = []                                                                                                                   # Create an empty list to store summarized pages.
        for e, _ in enumerate(pages):                                                                                                           # Iterate through the input pages with an index.
            if e > len(pages) - 1:                                                                                                              # If the index exceeds the number of pages, break out of the loop.
                break
            chunked_pages.append(pages[e:e+4])                                                                                                  # Chunk the pages into groups of 4 and append them to the list of chunked pages.
            e += 4                                                                                                                              # Increment the index by 4 for the next iteration.
        for page in chunked_pages:                                                                                                              # Iterate through the chunked pages.
            docs = [Document(page_content=page)]                                                                                                # Create a list of Document objects, each containing a chunked page of text.
            chain = load_summarize_chain(LLM, chain_type="map_reduce", return_intermediate_steps=False, map_prompt=PROMPT, combine_prompt=PROMPT)  # Load a summarization chain with the specified parameters, using the initialized OpenAI instance and the defined prompt template.
            result = chain({"input_documents": docs}, return_only_outputs=True)                                                                 # Execute the summarization chain with the input documents, and retrieve the output text.
            summarized_pages.append(result["output_text"])                                                                                      # Append the generated summary text to the list of summarized pages.
        get_summary(summarized_pages)                                                                                                           # Recursively call the get_summary function with the summarized pages as input, to continue summarizing until all pages are summarized.

