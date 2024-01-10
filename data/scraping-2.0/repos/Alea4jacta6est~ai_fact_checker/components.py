from langchain import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import (CharacterTextSplitter,
                                     RecursiveCharacterTextSplitter)

from ai_core import AIHandler

template = """Given this claim: {claim}
And these source paragraphs: {search_results}
Could you list only the factual errors and serious inconsistencies in the provided claim given these sources? Note, that the claim can be correct, return "Good" in this case
"""


def form_final_prompt(text, top_paragraphs):
    prompt = PromptTemplate.from_template(template)
    input_prompt = prompt.format(claim=text, search_results="\n".join(top_paragraphs))
    return input_prompt


def get_fact_checking(text: str, k=5):
    ai = AIHandler()
    source_chunks, _ = get_paragraphs(f"data/....")
    vectordb = ai.init_vectordb(source_chunks)
    result = vectordb.similarity_search(text, k)
    top_paragraphs = [result[i].dict()["page_content"] for i in range(len(result))]
    input_prompt = form_final_prompt(text, top_paragraphs)
    comment = ai.get_model_output(input_prompt)
    return comment


def get_paragraphs(path):
    doc = PyPDFLoader(path).load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    splits = text_splitter.split_documents(doc)
    chunks = list(set([item.page_content for item in splits]))
    return splits, chunks
