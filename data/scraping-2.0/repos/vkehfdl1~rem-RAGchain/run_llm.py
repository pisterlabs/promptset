from operator import itemgetter
from typing import List

import chromadb
import click
from RAGchain.reranker.time import WeightedTimeReranker
from RAGchain.retrieval import VectorDBRetrieval, HybridRetrieval, BM25Retrieval
from RAGchain.schema import Passage, RAGchainChatPromptTemplate
from RAGchain.utils.embed import EmbeddingFactory
from RAGchain.utils.vectorstore import ChromaSlim
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

CHROMA_DB_PATH = 'Chroma/'
BM25_PATH = 'DB/bm25.pkl'
PROMPT = RAGchainChatPromptTemplate.from_messages([
    ("user", "Please answer {question}"),
    ("system", "--------------- Below is the text that\'s been on screen recently. --------------"),
    ("system", "{passages}"),
    ("system", "Please answer the query using the provided information about what has been on the scren recently: "
               "{question}\nDo not say anything else or give any other information. Only answer the question : "
               "{question}"),
])

vectordb = ChromaSlim(
    client=chromadb.PersistentClient(path=CHROMA_DB_PATH),
    collection_name='rem',
    embedding_function=EmbeddingFactory('openai', device_type='mps').get()
)
retrieval = HybridRetrieval([VectorDBRetrieval(vectordb), BM25Retrieval(BM25_PATH)], [0.7, 0.3],
                            method='cc')
time_reranker = WeightedTimeReranker()


@click.command()
@click.option('--query', help='Query you want to ask to LLM.')
def main(query):
    print(f"Question : {query}")

    runnable = RunnablePassthrough.assign(
        passages=itemgetter("question") | RunnableLambda(
            lambda question: retrieve(question)),
        question=itemgetter("question"),
    ) | RunnablePassthrough.assign(
        answer={
                   "question": itemgetter("question"),
                   "passages": itemgetter("passages") | RunnableLambda(
                       lambda passages: Passage.make_prompts(passages),
                   ),
               } | PROMPT | ChatOpenAI() | StrOutputParser()
    )
    answer = runnable.invoke({"question": query})

    print(f"Answer : {answer['answer']}")
    print(f"Passages : {Passage.make_prompts(answer['passages'])}")


def retrieve(query: str, top_k: int = 4) -> List[Passage]:
    ids, scores = retrieval.retrieve_id_with_scores(query, top_k=100)
    passages = retrieval.fetch_data(ids)
    reranked_passages_and_scores = time_reranker.rerank(passages, scores)
    reranked_passages = [passage for passage, score in reranked_passages_and_scores]
    return reranked_passages[:top_k]


if __name__ == '__main__':
    main()
