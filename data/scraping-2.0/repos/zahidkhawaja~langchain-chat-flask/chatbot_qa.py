import json
import pathlib
from typing import Dict, List, Tuple
from langchain import OpenAI, PromptTemplate
from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Weaviate
from pydantic import BaseModel
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Weaviate, FAISS
from langchain.embeddings import OpenAIEmbeddings

_eg_template = """## Example:

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question: {answer}"""
_eg_prompt = PromptTemplate(
    template=_eg_template,
    input_variables=["chat_history", "question", "answer"],
)


_prefix= """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. You should assume that the question is related to LangChain."""
_suffix = """## Example:

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
WEAVIATE_URL = "https://hwc-testing.semi.network"
import weaviate
import os
client = weaviate.Client(
    url=WEAVIATE_URL,
    additional_headers={"X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]},
)
eg_store = Weaviate(client, "Rephrase", "content",
    attributes=["question", "answer", "chat_history"],)
example_selector = SemanticSimilarityExampleSelector(
    vectorstore=eg_store,
    k=4
)
prompt = FewShotPromptTemplate(prefix=_prefix, suffix=_suffix, example_selector=example_selector, example_prompt=_eg_prompt, input_variables=["question", "chat_history"])
llm = OpenAI(temperature=0,model_name="text-davinci-003")
key_word_extractor = LLMChain(llm=llm, prompt=prompt)


class CustomChain(Chain, BaseModel):

    vstore: Weaviate
    chain: BaseCombineDocumentsChain

    @property
    def input_keys(self) -> List[str]:
        return ["question"]

    @property
    def output_keys(self) -> List[str]:
        return ["answer"]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        question = inputs["question"]
        history = inputs["chat_history"]
        if history:
            new_question = key_word_extractor.run(question=question, chat_history=history)
        else:
            new_question = question
        print(new_question)
        docs = self.vstore.similarity_search(new_question, k=4)
        new_inputs = inputs.copy()
        new_inputs["question"] = new_question
        answer, _ = self.chain.combine_docs(docs, **new_inputs)
        return {"answer": answer}




def get_new_chain(vectorstore) -> Chain:
    eg_store = Weaviate(client, "QA", "content",
                        attributes=["question", "answer", "summaries", "sources"], )
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=eg_store,
        k=4
    )
    _eg_template = """## Example:

ONLY USE THE FOLLOWING PIECES OF INFORMATION
=========
{summaries}
=========
Question: {question}
Conversational Answer: {answer}
SOURCES: {sources}"""
    _eg_prompt = PromptTemplate(
        template=_eg_template,
        input_variables=["sources", "question", "answer", "summaries"],
    )
    prefix = """You are an AI assistant trained to have conversations to answer questions for the open source library LangChain.
Given the following extracted parts of a long document and a question, respond first with a conversational answer, and provide links to any relevant sources.
Remember, you are customer support bot for LangChain. If their question does not seem related to LangChain, please inform them nicely that you can only answer questions about LangChain.
If you are including a code snippet in your response, please do it in a way where it is rendered as a code block in markdown. If you are providing a code snippet, ONLY provide the input - do NOT attempt to complete it with what the output could be.
If you use information from ANY part of a source in your Conversational Answer, you MUST list it as a source in your response. Do not list sources that you did not use. Sources should be listed as a comma separated list.
In your Conversational Response, please do not link to sources directly. Rather, tell them to look at sources below, and then include that link in your Sources.
Here are some examples:"""

    suffix="""## Example

ONLY USE THE FOLLOWING PIECES OF INFORMATION
=========
{summaries}
=========
Question: {question}
Conversational Answer:"""
    PROMPT = FewShotPromptTemplate(
        prefix=prefix, suffix=suffix, example_selector=example_selector, example_prompt=_eg_prompt, input_variables=["summaries", "question"]
    )
    EXAMPLE_PROMPT = PromptTemplate(
        template=">Example:\nContent:\n---------\n{page_content}\n----------\nSource: {source}",
        input_variables=["page_content", "source"],
    )
    template = """You are an AI assistant for the open source library LangChain.
Given the following extracted parts of a long document and a question, create a final answer in Markdown.
The reference should be a full URL using the base URL https://langchain.readthedocs.io/en/latest/ and the path to the document.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not about LangChain, politely inform them that you are tuned to only answer questions about LangChain.

Question: {question}
=========
{context}
=========
Final answer in Markdown:"""
    PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])
    doc_chain = load_qa_chain(
        OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=-1),
        chain_type="stuff",
        prompt=PROMPT,
        document_prompt=EXAMPLE_PROMPT
    )
    # TODO: update once new version
    # doc_chain.document_prompt = EXAMPLE_PROMPT
    memory = ConversationBufferMemory(
        memory_key="chat_history", output_key="answer", ai_prefix="Assistant"
    )
    return CustomChain(chain=doc_chain, vstore=vectorstore, memory=memory)

def get_new_chain1(vectorstore) -> Chain:
    _eg_template = """## Example:

ONLY USE THE FOLLOWING PIECES OF INFORMATION
=========
{summaries}
=========
Question: {question}
Conversational Answer: {answer}
SOURCES: {sources}"""
    _eg_prompt = PromptTemplate(
        template=_eg_template,
        input_variables=["sources", "question", "answer", "summaries"],
    )


    EXAMPLE_PROMPT = PromptTemplate(
        template=">Example:\nContent:\n---------\n{page_content}\n----------\nSource: {source}",
        input_variables=["page_content", "source"],
    )
#     template = """You are an AI assistant for the open source library LangChain. The documentation is located at https://langchain.readthedocs.io.
# You are given the following extracted parts of a long document and a question. Provide a conversational answer with a hyperlink to the documentation.
# If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
# If the question is not about LangChain, politely inform them that you are tuned to only answer questions about LangChain.
# Question: {question}
# =========
# {context}
# =========
# Answer in Markdown:"""
#     template = """You are an AI assistant for the open source library LangChain. The documentation is located at https://langchain.readthedocs.io.
# You are given the following extracted parts of a long document and a question. Provide a helpful and concise answer.
# Your answer must include at least 1 relevant link to the documentation. When appropriate, provide useful code samples to help the developer.
# If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
# If the question is not about LangChain, politely inform them that you are tuned to only answer questions about LangChain.
# Question: {question}
# =========
# {context}
# =========
# Answer in Markdown:"""
#     template = """You are an AI assistant for the open source library LangChain. The documentation is located at https://langchain.readthedocs.io.
# You are given the following extracted parts of a long document and a question. Provide a helpful and concise answer.
# Your answer should include only 1 hyperlink to the documentation. Provide code samples directly from the documentation when necessary.
# If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
# If the question is not about LangChain, politely inform them that you are tuned to only answer questions about LangChain.
# Question: {question}
# =========
# {context}
# =========
# Answer in Markdown:"""
    template = """You are an AI assistant for the open source library LangChain. The documentation is located at https://langchain.readthedocs.io.
You are given the following extracted parts of a long document and a question. Provide a conversational answer with a hyperlink to the documentation.
You should only use hyperlinks that are explicitly listed as a source in the context. Do NOT make up a hyperlink that is not listed.
If the question includes a request for code, provide a code block directly from the documentation.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not about LangChain, politely inform them that you are tuned to only answer questions about LangChain.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""
    PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])
    doc_chain = load_qa_chain(
        OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=-1),
        chain_type="stuff",
        prompt=PROMPT,
        document_prompt=EXAMPLE_PROMPT
    )
    return CustomChain(chain=doc_chain, vstore=vectorstore)


def _get_chat_history(chat_history: List[Tuple[str, str]]):
    buffer = ""
    for human_s, ai_s in chat_history:
        human = f"Human: " + human_s
        ai = f"Assistant: " + ai_s
        buffer += "\n" + "\n".join([human, ai])
    return buffer

import os

import weaviate
from langchain.vectorstores import Weaviate

WEAVIATE_URL = "https://hwc-testing.semi.network"


def get_weaviate_store():
    client = weaviate.Client(
        url=WEAVIATE_URL,
        additional_headers={"X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]},
    )
    return Weaviate(client, "Paragraph", "content", attributes=["source"])

vectorstore = get_weaviate_store()
CHAIN = get_new_chain1(vectorstore)


def new_predict(question: str, chat_history: List[Tuple[str, str]]):
    chat_history_str = _get_chat_history(chat_history)
    return CHAIN.run(question=question, chat_history=chat_history_str)