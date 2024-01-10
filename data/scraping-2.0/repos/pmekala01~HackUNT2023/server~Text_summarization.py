from langchain.llms.openai import OpenAI
from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
import os
from langchain.prompts import PromptTemplate
import tiktoken

def num_tokens_from_string(string, encoding_name):
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

load_dotenv()

def Summarization(text, openai_api_key = os.getenv("OPENAI_API_KEY")):
    # Creating the llm object with the open ai key.
    llm = OpenAI(temperature=0, openai_api_key = openai_api_key)
    model_name = "gpt-3.5-turbo"
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(model_name = model_name)
    
    texts = text_splitter.split_text(text)
    text = [Document(page_content=t) for t in texts]

    # Creating the prompt template.
    prompt_template = """Write a concise summary of the following:

    {text}
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    combined_text = "\n".join(texts)
    num_tokens = num_tokens_from_string(combined_text, model_name)
    gpt_35_turbo_max_tokens = 4097
    verbose = True

    if num_tokens < gpt_35_turbo_max_tokens:
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt, verbose=verbose)
    else:
        chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=prompt, combine_prompt=prompt, verbose=verbose)

    summary = chain.run(text)
    return summary
