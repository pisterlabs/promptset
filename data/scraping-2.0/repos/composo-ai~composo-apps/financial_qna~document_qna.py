from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import openai
import tiktoken

db = FAISS.load_local("embeddings/2022-alphabet-annual-report", OpenAIEmbeddings())

default_system_message = "You are a helpful customer service assistant. Use the snippets provided to answer the user's question: {retrieved_snippets}"


def document_qa(
    temperature: float,
    system_message: str,
    retrieval_count: int,
    query: str,
):
    if len(tiktoken.encoding_for_model("gpt-3.5-turbo").encode(query)) > 256:
        return "This demo app has a 256 token limit for queries"

    if len(tiktoken.encoding_for_model("gpt-3.5-turbo").encode(system_message)) > 256:
        return "This demo app has a 256 token limit for system messages"

    docs = db.similarity_search(query, k=retrieval_count)

    retrieved_snippets = (
        "\n```snippet\n"
        + "\n```\n```snippet\n".join(x.page_content for x in docs)
        + "\n```"
    )

    messages = [
        {
            "role": "system",
            "content": system_message.format(retrieved_snippets=retrieved_snippets),
        },
        {"role": "user", "content": query},
    ]

    return (
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=temperature,
            max_tokens=256,
        )
        .choices[0]
        .message["content"]
    )


answer = document_qa(
    temperature=1,
    system_message=default_system_message,
    retrieval_count=3,
    query="How is Google helping people find new ways to empower people to stay on top of their wellness",
)
print(answer)
