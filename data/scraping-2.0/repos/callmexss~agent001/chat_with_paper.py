from codecs import ignore_errors
import logging
from pathlib import Path

import openai
import pandas as pd
import rich
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from langchain.vectorstores import Chroma
from rich.logging import RichHandler

from agent import settings
from agent.tools.ingest_doc import ingest_pdf

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])


QUESTIONS = [
    "这篇论文的主题是什么？论文的主题是否清晰易懂？",
    "这篇论文的主要观点或论断是什么？作者的论述是否有足够的支撑？",
    "这篇论文的理论框架是什么？这个框架是否有效地支持了论文的主题和主要观点？",
    "这篇论文的方法论或研究方法是什么？这些方法是否可靠、科学、合理？",
    "这篇论文的结果或结论是否支持其主要论点？这些结果是否能够经得起批判性的审查？",
    "这篇论文的新颖性或创新性在哪里？是否对该领域的研究提供了新的洞见或贡献？",
    "这篇论文的限制或不足在哪里？作者是否明确地表述了这些限制？",
    "作者的引用和参考文献是否充足、恰当？这些引用是否支持其论点？",
    "这篇论文的结构和逻辑是否清晰？是否有足够的过渡句和段落来引导读者？",
    "总体而言，这篇论文的质量如何？是否值得进一步研究或参考？",
]


system_template = """
Consider the context provided to answer the user's question accurately.
If the answer is beyond your knowledge, it's better to admit that you don't know instead of fabricating a response.
Whenever possible, provide a reliable reference to support your answer.
Aim to explain concepts in a way that a 7-year-old would understand, keeping language simple and using examples when appropriate.
At the end of your response, please generate five insightful questions related to the context that could stimulate further discussion or thought.
----------------
{context}
----------------
"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)


def ask(query: str, chain):
    result = chain({"query": query})
    print()
    with open(
        settings.CHAT_HISTORY_PATH / f"chat_history_{name}.txt",
        "a+",
        encoding="utf-8",
        errors="ignore"
    ) as f:
        f.write("==========================================================\n")
        f.write(f"Q: {query}\nA: {result['result']}\n\n")
        f.write(
            "\nC: "
            + "\n\n------\n\n".join(
                [page.page_content for page in result["source_documents"]]
            )
        )
        f.write("\n==========================================================\n\n")
    return result


def chat(chain):
    while True:
        try:
            query = input("Q: ")
            if query == "exit":
                break
            result = ask(query, chain)
            print('\n', '*' * 80, '\n')
            rich.print(result['result'])
        except KeyboardInterrupt:
            break
        except openai.error.OpenAIError as e:
            logging.info(f"OpenAI API error: {e}")
        except openai.error.InvalidRequestError as e:
            logging.info(f"Invalid request error: {e}")
        except UnicodeEncodeError as e:
            logging.info(f"Unicode decode error: {e}")


def auto_ask(chain):
    for question in QUESTIONS:
        _ = ask(question, chain)


if __name__ == "__main__":
    df = pd.read_parquet(".vector_db/chroma-collections.parquet")
    name_li = df["name"].to_list()
    rich.print([f"{i}. {name}" for i, name in enumerate(name_li)])
    idx = input("Which paper do you want to explore? ")
    name = name_li[max(0, min(int(idx), len(name_li)))]

    # for name in name_li:
    #     db = Chroma(
    #         collection_name=name,
    #         embedding_function=OpenAIEmbeddings(),
    #         persist_directory=settings.DB_PATH.as_posix(),
    #     )
    #     db.persist()

    db = Chroma(
        collection_name=name,
        embedding_function=OpenAIEmbeddings(),
        persist_directory=settings.DB_PATH.as_posix(),
    )

    llm = ChatOpenAI(
        temperature=0.2,
        # model="gpt-3.5-turbo-16k-0613",
        # model="gpt-3.5-turbo-0613",
        model="gpt-4-0613",
        streaming=True,
        max_tokens=1000,
        callbacks=[StreamingStdOutCallbackHandler()],
    )

    llm_16k = ChatOpenAI(
        temperature=0.2,
        model="gpt-3.5-turbo-16k-0613",
        streaming=True,
        max_tokens=2000,
        callbacks=[StreamingStdOutCallbackHandler()],
    )
    retriever = db.as_retriever()
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 100
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 10

    chain = RetrievalQA.from_llm(
        llm_16k,
        retriever=retriever,
        callbacks=[StreamingStdOutCallbackHandler()],
        return_source_documents=True,
        prompt=CHAT_PROMPT,
        memory=ConversationBufferMemory(
            memory_key="chat_history", input_key="query", output_key="result"
        ),
    )

    # auto_ask(chain)
    chat(chain)

    # papers = Path("./data/papers").glob("*.pdf")
    # for paper in papers:
    #     if paper.name[:4] not in name_li:
    #         print(paper.absolute().as_posix())
    #         ingest_pdf(paper, chunk_size=2000, chunk_overlap=100)
