#!/usr/bin/env python
# coding: utf-8
import os

from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI, PromptTemplate
from translate import translate_with_chatgpt

import utils
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())


# summarize with langchain
def summarize_with_langchain(path, filename):
    # Load your documents
    with open(path + filename, encoding="utf-8") as f:
        text = f.read()
    new_text = utils.extract_text_from_subtitle(text)
    # Get your splitter ready
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=20)

    # Split your docs into texts
    texts = text_splitter.split_text(new_text)
    from langchain.docstore.document import Document

    docs = [Document(page_content=t) for t in texts]

    # There is a lot of complexity hidden in this one line. I encourage you to check out the video above for more detail
    llm = OpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"),
                 openai_api_base='https://api.openai-proxy.com/v1')
    user_prompt = """
Task: 
    1. Summarization: Generate a concise summary of the Text at most 125 words. 
    2. Topics: Generate 5 main topics of the Text.
    3. Tags: Generate 5 tags of the Text for youtube recommend.
The output markdown format is as follows:
## Summarization:
## Topics: 
    1. abc
    2. dfe 
    3. ...
## Tags: 
    aaaa,bbbb,cddd,deeea
Text: ```{text}```
    """
    print(user_prompt)
    PROMPT = PromptTemplate(template=user_prompt, input_variables=["text"])
    if len(user_prompt) + len(text) < 4000:
        chain = load_summarize_chain(llm, chain_type="stuff", verbose=True,
                                     prompt=PROMPT)
    else:
        chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True,
                                     combine_prompt=PROMPT)
    ans_text = chain.run(docs)

    # 输出结果
    return ans_text


if __name__ == '__main__':
    o = summarize_with_langchain('./data/',
                                 'Build Your Own Auto-GPT Apps with LangChain (Python Tutorial) [NYSWn1ipbgg].txt')
    t = translate_with_chatgpt(o, 'zh')
    print(o)
    print(t)
