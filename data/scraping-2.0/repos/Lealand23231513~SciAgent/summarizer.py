from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.prompts import PromptTemplate
from search_and_download import download_arxiv_pdf
import os
import json
import openai
import logging
from pathlib import Path
logger = logging.getLogger(Path(__file__).stem)
# DEFAULT_PATH = "./.cache"

def summary(path:str):
    '''
    :param: path: path of the file.
    '''
    if(path.split(".")[-1] == 'pdf'):
        loader = PyPDFLoader(path)
    elif(path.split(".")[-1] == 'docx'):
        loader = Docx2txtLoader(path)
    else:
        logger.warning("document not found")
        return None
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    logger.debug(f'documents:{len(docs)}')

    prompt_template = """Write a summary of this paper,
    which should contain introduction of research field, process and achievements:
 
    {text}
 
    SUMMARY Here:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

    llm = OpenAI(temperature=0.2, max_tokens=1000, model="gpt-3.5-turbo-instruct")
    logger.debug(llm.model_name)
    chain = load_summarize_chain(llm, chain_type="map_reduce", return_intermediate_steps=False, map_prompt=PROMPT, combine_prompt=PROMPT)
    summary_result = chain({"input_documents": docs}, return_only_outputs=True)["output_text"]
    return summary_result
def summarizer(papers_info):
    ai_response = []
    for i,paper_info in enumerate(papers_info):
        file_path = download_arxiv_pdf(paper_info)
        papers_info[i]["path"] = file_path
        summary_result = summary(file_path)
        ai_response.append(f"Succesfully download <{paper_info['Title']}> into {file_path} !\n The summary result is as below:\n{summary_result}")
    return "\n".join(ai_response)
    
if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    test_file = "TEST  Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series.pdf"
    summary_result = summary(test_file)
    logger.info(summary_result)
    #summary("C:\Pythonfiles\langchain_try\summary\\test_paper\Attention Is All You Need.pdf")