"""
ask_questions.py

Functions for answering questions based on the embeddings and knowledge graph created.
"""

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from tqdm import tqdm

import graph_handling
import embedding_handling


def get_answer_from_sources(
    question: str,
    relationship_summaries: list[str],
    document_chunks: list[str],
    llm: ChatOpenAI = graph_handling.GPT4_TURBO,
) -> str:
    if len(relationship_summaries) == 0 and len(document_chunks) == 0:
        return "Sorry, but I couldn't find enough information to answer your question."

    formatted_summaries = "\n\n".join(relationship_summaries)
    formatted_chunks = "\n\n".join(document_chunks)

    messages = [
        SystemMessage(
            content=f"""You are a helpful assistant that responds to user queries given the following context. You are given both a series of facts from a knowledge graph, and raw chunks of documents. Combine both sources to provide a truthful answer.
            
            Here is a collection of facts pulled from a knowledge graph:

            {formatted_summaries}

            Here is a collection of relevant chunks of notes, pulled from a collection of the user's documents:

            {formatted_chunks}

            Use all of the above resources to answer the user's question in as much detail as the context can provide by itself. If the provided facts or document chunks do not provide enough information to answer the question, then say "Sorry, but I could not find enough information to answer the question."
            """
        ),
        HumanMessage(content=question),
    ]

    return llm(messages).content


def answer_question(
    question: str,
    llm: ChatOpenAI = graph_handling.GPT4_TURBO,
) -> str:
    """
    Answers a question based on the embeddings and knowledge graph created from the
    Obsidian vault.

    This pulls in information from both the embeddings and the knowledge graph:

    For the embeddings:
    - Uses GPT to convert the question into a declarative statement
    - Creates embeddings from the declarative statement
    - Does a search of the vector store and grabs the most relevant document chunk IDs
      by cosine similarity

    For the knowledge graph:
    - Uses GPT to identify which nodes in the graph are most relevant to the question
    - Collects the relationships between these nodes in the graph and has GPT summarize
      these relationships
    - Collects the IDs of the document chunks that reference the most relevant nodes

    Together:
    - Combines the collected document chunk IDs from the embeddings and knowledge graph,
      and queries the vector store to collect the associated document chunks
    - Sends GPT a prompt containing the relationship summaries and the retrieved
      document chunks, alongside the user's original question.
    """

    result = ""

    with tqdm(total=7) as pbar:
        print(
            "Converting user query into declarative statement, creating embeddings, and searching the vector store for similar document chunks."
        )
        vector_search_result = embedding_handling.user_query_to_chromadb_query_result(
            question
        )
        pbar.update(1)

        print("Collecting most relevant document chunks by embedding similarity.")
        chunk_ids_from_embeddings = embedding_handling.chroma_query_result_to_chunk_ids(
            vector_search_result
        )
        pbar.update(1)

        print("Identifying most relevant nodes in knowledge graph.")
        node_names = graph_handling.get_relevant_nodes_from_question(
            graph_handling.GPT4, graph_handling.get_all_node_names(), question
        ).names
        pbar.update(1)

        print("Collecting relevant document chunks by identified KG nodes.")
        chunk_ids_from_graph = graph_handling.get_chunk_ids_by_node_names(node_names)
        pbar.update(1)

        # Combine the chunk IDs
        combined_chunk_ids = list(set(chunk_ids_from_embeddings + chunk_ids_from_graph))

        # Grab vector store connection
        collection = embedding_handling.get_vector_store_collection()

        # Grab document chunks by IDs
        docs = collection.get(ids=combined_chunk_ids, include=["documents"])[
            "documents"
        ]

        print("Collecting relationships from knowledge graph between identified nodes.")
        node_relationships = graph_handling.get_interrelationships_between_nodes(
            node_names
        )
        pbar.update(1)

        print("Summarizing knowledge graph relationships.")
        node_relationship_summaries = list(
            map(
                lambda r: graph_handling.summarize_relationship(
                    graph_handling.GPT4_TURBO, r
                ),
                node_relationships,
            )
        )
        pbar.update(1)

        print(
            "Providing relationships and document chunks to GPT to answer the question."
        )
        result = get_answer_from_sources(
            question, node_relationship_summaries, docs, llm
        )
        pbar.update(1)

    return result
