"""Langchain chains"""
import os
from typing import List
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import SequentialChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import (
    ReduceDocumentsChain,
    MapReduceDocumentsChain,
    StuffDocumentsChain,
)
from langchain.callbacks import get_openai_callback
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import PlaywrightURLLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.llms.openai import OpenAI

# from redis import Redis
# from langchain.cache import RedisCache
import streamlit as st

os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
# langchain.llm_cache = RedisCache(redis_=Redis(host=st.secrets["redis"]["host"],
#                                              port=st.secrets["redis"]["port"], db=0))


def question_to_words_chain(llm):
    # Prompt
    template = """
    You will be given a question.
    Your goal is to transform the question to 3 words maximum. Remove the question mark.
    {format_instructions}

    <question>
    {question}
    </question>
    words:"""
    _format = """
    ```
    words:transformed question
    ...
    ```
    e.g of words:
    ```
    <question>
    What are the intellectual property rights?
    </question>
    words:Intellectual Property Rights
    ```"""
    prompt = PromptTemplate(
        input_variables=["question"],
        partial_variables={"format_instructions": _format},
        template=template,
    )
    return LLMChain(llm=llm, prompt=prompt, output_key="words", verbose=True)


def words_to_emoji(llm):
    # Prompt
    template = """
    You will be given a set of words.
    Your goal is to understand the sequence of words and choose the best emoji to describe it.
    You should only output an emoji.
    {format_instructions}

    <words>
    {words}
    </words>
    emoji:"""
    _format = """
    ```
    emoji:emoji best describing the words
    ...
    ```
    e.g of emoji:
    ```
    <words>
    Intellectual Property Rights
    </words>
    emoji:👨‍💼
    ```"""

    prompt = PromptTemplate(
        input_variables=["words"],
        partial_variables={"format_instructions": _format},
        template=template,
    )
    return LLMChain(llm=llm, prompt=prompt, output_key="emoji", verbose=True)


# Define your desired data structure.
class TermsAnswer(BaseModel):
    answer: str = Field(description="Answer of the question")
    excerpts: str = Field(description="Phrase from terms that justify the answer")


def answer_question_chain(llm):
    # Prompt
    template = """
    Your goal is to read a term of use and answer questions about it.
    Additionnaly, provide excerpts of the text that justify your answer.
    {format_instructions}

    <terms>
    {terms}
    </terms>
    <question>
    {question}
    </question>
    output:"""

    # Set up a parser + inject instructions into the prompt template.
    parser = PydanticOutputParser(pydantic_object=TermsAnswer)

    prompt = PromptTemplate(
        input_variables=["terms", "question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
        template=template,
    )
    return LLMChain(llm=llm, prompt=prompt, output_key="output", verbose=True)


def overall_chain_exec(questions: list, terms: str):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
    llm2 = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    answers = {}

    seq_chain = SequentialChain(
        chains=[
            question_to_words_chain(llm2),
            words_to_emoji(llm2),
            answer_question_chain(llm),
        ],
        input_variables=["terms", "question"],
        output_variables=["words", "emoji", "output"],
        verbose=True,
    )
    for question in questions:
        term_answer = seq_chain(
            {"terms": terms, "question": question}, return_only_outputs=True
        )

        # Add the answer to the dictionary
        parser = PydanticOutputParser(pydantic_object=TermsAnswer)
        answers[question] = {
            "emoji": term_answer["emoji"],
            "words": term_answer["words"],
            "answer": parser.parse(term_answer["output"]).answer,
            "excerpts": parser.parse(term_answer["output"]).excerpts,
        }
    return answers


def overall_chain_url_exec(questions: list, terms_url: str):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
    llm2 = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    answers = {}

    seq_chain = SequentialChain(
        chains=[
            question_to_words_chain(llm2),
            words_to_emoji(llm2),
            answer_question_chain(llm),
        ],
        input_variables=["terms", "question"],
        output_variables=["words", "emoji", "output"],
        verbose=True,
    )
    loader = PlaywrightURLLoader(
        urls=[terms_url], remove_selectors=["header", "footer"]
    )
    docs = loader.load()

    for question in questions:
        term_answer = seq_chain(
            {
                "terms": " ".join([doc.page_content for doc in docs]),
                "question": question,
            },
            return_only_outputs=True,
        )

        # Add the answer to the dictionary
        parser = PydanticOutputParser(pydantic_object=TermsAnswer)
        answers[question] = {
            "emoji": term_answer["emoji"],
            "words": term_answer["words"],
            "answer": parser.parse(term_answer["output"]).answer,
            "excerpts": parser.parse(term_answer["output"]).excerpts,
        }
    return answers


def map_reduce_summarization_chain(llm):
    # Reduce
    reduce_template = """The following is set of summaries:
    {doc_summaries}
    Take these and distill it into a final, consolidated summary.
    Remember the summaries comes from term of use and general policy of a website or application, so it's not necessary to mention it.
    Helpful Answer:"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    # Run chain
    return LLMChain(llm=llm, prompt=reduce_prompt)


def map_chain(llm):
    map_template = """The following is a set of documents that represents the term of use and general policy of a website or application:
    {docs}
    Based on this list of docs, please identify the main themes.
    Remember the summaries comes from term of use and general policy of a website or application, so it's not necessary to mention it.
    Helpful Answer:"""
    map_prompt = PromptTemplate.from_template(map_template)
    return LLMChain(llm=llm, prompt=map_prompt)


def overall_summarize_chain_exec(terms: str):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=map_reduce_summarization_chain(llm),
        document_variable_name="doc_summaries",
    )

    # Combines and iteravely reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=15000,
    )

    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain(llm),
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="docs",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=0, separators=[" ", ",", "\n"]
    )
    docs = [Document(page_content=terms)]
    split_docs = text_splitter.split_documents(docs)

    return map_reduce_chain.run(split_docs)


def overall_summarize_chain_url_exec(terms_url: str):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=map_reduce_summarization_chain(llm),
        document_variable_name="doc_summaries",
    )

    # Combines and iteravely reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=15000,
    )

    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain(llm),
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="docs",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=0, separators=[" ", ",", "\n"]
    )

    loader = WebBaseLoader(terms_url)
    docs = loader.load()
    split_docs = text_splitter.split_documents(docs)

    return map_reduce_chain.run(split_docs)


def summarize_chain_exec(terms: str):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    docs = [Document(page_content=terms)]
    return chain.run(docs)


def summarize_chain_doc_exec(terms: List[Document]):
    llm = OpenAI(temperature=0, model="gpt-3.5-turbo-instruct")
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    with get_openai_callback() as cb:
        result = chain.run(terms)
    return cb, result


def summarize_chain_url_exec(terms_url: str):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")

    prompt_template = """The following is set of summaries:

    {text}

    Take these and distill it into a final, consolidated summary.
    Remember the summaries comes from term of use and general policy of a website or application, so it's not necessary to mention it.
    CONCISE SUMMARY:"""
    combine_prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    prompt_template = """The following is a set of documents that represents the term of use and general policy of a website or application.
    The documents are parsed from an html page and may contain some elements unrelated to the term of use, ignore them:

    {text}

    Based on this list of documents, please summaries them and highlights the key points that a user of the website or application might be interested.
    Remember the summaries comes from term of use and general policy of a website or application, so it's not necessary to mention it.
    CONCISE SUMMARY:"""
    map_prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
    )
    loader = PlaywrightURLLoader(
        urls=[terms_url], remove_selectors=["header", "footer"]
    )
    print("load")
    docs = loader.load()
    print("loaded")
    if docs and len(docs) > 0:
        return chain.run(docs)
    return "No data"


def simple_qa_chain(query, docsearch: VectorStoreRetriever, language="English"):
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer with 2 to 3 words in the same language of the question:"""
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Get the 3 most relevant documents
    docs = docsearch.get_relevant_documents(query)[:3]
    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff", prompt=prompt)
    return docs, chain(
        {"input_documents": docs, "question": query},
        return_only_outputs=True,
    )


def simple_qa_chain_long(query, docs):
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer and give an excerpt to validate the answer in {language}:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Get the 3 most relevant documents
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
    chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)
    return docs, chain(
        {"input_documents": docs, "question": query, "language": "English"},
        return_only_outputs=True,
    )
