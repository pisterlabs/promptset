# openai api
import openai

# langchain module
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks import get_openai_callback

# pdf loader
from langchain.document_loaders import PyPDFLoader, ArxivLoader
# url loader
from langchain.document_loaders import UnstructuredURLLoader
# youtube loader
from langchain.document_loaders import YoutubeLoader

# time module
from tqdm import tqdm

import re


from langchain.schema.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

def translate(config):
    
    if config.pdf:
        system = """Please translate arxiv paper with Markdown format using header.
        you Must translate into Korean.
        Paper:\n\n{text}"""  # [5]
    if config.html:
        system = """You can use the following template.\n
        1. Output is Markdown format.\n
        2. Organize the content of a web page and write it in markdown.\n
        3. You must translate into Korean.\n
        4. below triple bracket is web page url.\n
        5. subtitle use markdown H4.\n
        6. title use markdown H3.\n
        7. if you want to use math formula, use latex.\n
        8. if you want to use image, use markdown image.\n
        9. if you want to use code, use markdown code.\n
        10. if you want to use table, use markdown table.\n"""

    if config.youtube:
        system = """You can use the following template.\n
        1. this is youtube script.\n
        2. translate into Korean.\n
        """
    # select llm model
    llm = ChatOpenAI(model=config.model)

    # usage token
    completion_tokens = 0
    prompt_tokens = 0
    total_cost = 0
    total_tokens = 0

    results = []
    for page in tqdm(config.document, total=len(config.document)):
        
        with get_openai_callback() as cb:
            result = llm([SystemMessage(content=system), HumanMessage(content=page.page_content)])
            print(result.content)

            prompt_tokens += cb.prompt_tokens
            total_cost += cb.total_cost
            total_tokens += cb.total_tokens
            results.append(result.content)
            
    print(f"completion_tokens: {completion_tokens}\n")
    print(f"prompt_tokens: {prompt_tokens}\n")
    print(f"total_cost: {total_cost}\n")
    print(f"total_tokens: {total_tokens}\n")

    for result in results:
        print(result)
    
    if config.outputfile:
        with open(config.outputfile, "w") as f:
            for result in results:
                f.writelines(result)

            f.writelines("\n# Cost\n")
            f.writelines(f"\n- completion_tokens: {completion_tokens}\n")
            f.writelines(f"- prompt_tokens: {prompt_tokens}\n")
            f.writelines(f"- total_cost: {total_cost}$\n")
            f.writelines(f"- total_tokens: {total_tokens}\n")
            
    print("Done!")
