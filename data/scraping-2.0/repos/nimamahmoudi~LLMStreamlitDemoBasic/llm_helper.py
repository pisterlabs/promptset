from typing import Optional

# langchain imports
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnableMap
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from operator import itemgetter
from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage
from langchain.callbacks.streamlit.streamlit_callback_handler import StreamlitCallbackHandler


def format_docs(docs):
    res = ""
    # res = str(docs)
    for doc in docs:
        escaped_page_content = doc.page_content.replace("\n", "\\n")
        res += "<doc>\n"
        res += f"  <content>{escaped_page_content}</content>\n"
        for m in doc.metadata:
            res += f"  <{m}>{doc.metadata[m]}</{m}>\n"
        res += "</doc>\n"
    return res


def get_search_index(file_name="Mahmoudi_Nima_202202_PhD.pdf", index_folder="index"):
    # load embeddings
    from langchain.vectorstores import FAISS
    from langchain.embeddings.openai import OpenAIEmbeddings

    search_index = FAISS.load_local(
        folder_path=index_folder,
        index_name=file_name + ".index",
        embeddings=OpenAIEmbeddings(),
    )
    return search_index


def convert_message(m):
    if m["role"] == "user":
        return HumanMessage(content=m["content"])
    elif m["role"] == "assistant":
        return AIMessage(content=m["content"])
    elif m["role"] == "system":
        return SystemMessage(content=m["content"])
    else:
        raise ValueError(f"Unknown role {m['role']}")


_condense_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {input}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_condense_template)

_rag_template = """Answer the question based only on the following context, citing the page number(s) of the document(s) you used to answer the question:
{context}

Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(_rag_template)


def _format_chat_history(chat_history):
    def format_single_chat_message(m):
        if type(m) is HumanMessage:
            return "Human: " + m.content
        elif type(m) is AIMessage:
            return "Assistant: " + m.content
        elif type(m) is SystemMessage:
            return "System: " + m.content
        else:
            raise ValueError(f"Unknown role {m['role']}")

    return "\n".join([format_single_chat_message(m) for m in chat_history])

def get_standalone_question_from_chat_history_chain():
    _inputs = RunnableMap(
        standalone_question=RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | ChatOpenAI(temperature=0)
        | StrOutputParser(),
    )
    return _inputs

def get_rag_chain(file_name="Mahmoudi_Nima_202202_PhD.pdf", index_folder="index", retrieval_cb=None):
    vectorstore = get_search_index(file_name, index_folder)
    retriever = vectorstore.as_retriever()

    if retrieval_cb is None:
        retrieval_cb = lambda x: x

    def context_update_fn(q):
        retrieval_cb([q])
        return q

    _inputs = RunnableMap(
        standalone_question=RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | ChatOpenAI(temperature=0)
        | StrOutputParser(),
    )
    _context = {
        "context": itemgetter("standalone_question") | RunnablePassthrough(context_update_fn) | retriever | format_docs,
        "question": lambda x: x["standalone_question"],
    }
    conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | ChatOpenAI()
    return conversational_qa_chain


# RAG fusion chain
# source1: https://youtu.be/GchC5WxeXGc?si=6i7J0rPZI7SNwFYZ
# source2: https://towardsdatascience.com/forget-rag-the-future-is-rag-fusion-1147298d8ad1
def reciprocal_rank_fusion(results: list[list], k=60):
    from langchain.load import dumps, loads
    fused_scores = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results


def get_search_query_generation_chain():
    from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
    prompt = ChatPromptTemplate(
        input_variables=['original_query'],
        messages=[
            SystemMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=[],
                    template='You are a helpful assistant that generates multiple search queries based on a single input query.'
                )
            ),
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=['original_query'],
                    template='Generate multiple search queries related to: {original_query} \n OUTPUT (4 queries):'
                )
            )
        ]
    )

    generate_queries = (
        prompt |
        ChatOpenAI(temperature=0) |
        StrOutputParser() |
        (lambda x: x.split("\n"))
    )

    return generate_queries

def get_rag_fusion_chain(file_name="Mahmoudi_Nima_202202_PhD.pdf", index_folder="index", retrieval_cb=None):
    vectorstore = get_search_index(file_name, index_folder)
    retriever = vectorstore.as_retriever()
    query_generation_chain = get_search_query_generation_chain()
    _inputs = RunnableMap(
        standalone_question=RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | ChatOpenAI(temperature=0)
        | StrOutputParser(),
    )

    if retrieval_cb is None:
        retrieval_cb = lambda x: x

    _context = {
        "context":
            RunnablePassthrough.assign(
                original_query=lambda x: x["standalone_question"]
            )
            | query_generation_chain
            | retrieval_cb
            | retriever.map()
            | reciprocal_rank_fusion
            | (lambda x: [item[0] for item in x])
            | format_docs,
        "question": lambda x: x["standalone_question"],
    }
    conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | ChatOpenAI()
    return conversational_qa_chain


####################
# Adding agent chain with OpenAI function calling

def get_search_tool_from_index(search_index, st_cb: Optional[StreamlitCallbackHandler] = None, ):
    from langchain.agents import tool
    from agent_helper import retry_and_streamlit_callback

    @tool
    @retry_and_streamlit_callback(st_cb=st_cb, tool_name="Content Seach Tool")
    def search(query: str) -> str:
        """Search the contents of the source document for the queries."""

        docs = search_index.similarity_search(query, k=5)
        return format_docs(docs)
    
    return search

def get_lc_oai_tools(file_name:str = "Mahmoudi_Nima_202202_PhD.pdf", index_folder: str = "index", st_cb: Optional[StreamlitCallbackHandler] = None, ):
    from langchain.tools.render import format_tool_to_openai_tool
    search_index = get_search_index(file_name, index_folder)
    lc_tools = [get_search_tool_from_index(search_index=search_index, st_cb=st_cb)]
    oai_tools = [format_tool_to_openai_tool(t) for t in lc_tools]
    return lc_tools, oai_tools

def get_agent_chain(file_name="Mahmoudi_Nima_202202_PhD.pdf", index_folder="index", callbacks=None, st_cb: Optional[StreamlitCallbackHandler] = None, ):
    if callbacks is None:
        callbacks = []

    from langchain.agents import initialize_agent, AgentType
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.agents.format_scratchpad.openai_tools import (
        format_to_openai_tool_messages,
    )
    from langchain.agents import AgentExecutor
    from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

    lc_tools, oai_tools = get_lc_oai_tools(file_name, index_folder, st_cb)
    

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant, use the search tool to answer the user's question and cite only the page number when you use information coming (like [p1]) from the source document.\nchat history: {chat_history}"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106")

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
            "chat_history": lambda x: _format_chat_history(x["chat_history"]),
        }
        | prompt
        | llm.bind(tools=oai_tools)
        | OpenAIToolsAgentOutputParser()
    )

    agent_executor = AgentExecutor(agent=agent, tools=lc_tools, verbose=True, callbacks=callbacks)
    return agent_executor


if __name__ == "__main__":
    question_generation_chain = get_search_query_generation_chain()
    print('='*50)
    print('RAG Chain')
    chain = get_rag_chain()
    print(chain.invoke({'input': 'serverless computing', 'chat_history': []}))

    print('='*50)
    print('Question Generation Chain')
    print(question_generation_chain.invoke({'original_query': 'serverless computing'}))

    print('-'*50)
    print('RAG Fusion Chain')
    chain = get_rag_fusion_chain()
    print(chain.invoke({'input': 'serverless computing', 'chat_history': []}))

    agent_executor = get_agent_chain()
    print(
        agent_executor.invoke({
            "input": "based on the source document, compare FaaS with BaaS??",
            "chat_history": [],
        })
    )
