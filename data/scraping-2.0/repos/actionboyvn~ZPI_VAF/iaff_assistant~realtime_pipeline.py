from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import tiktoken
import openai
from dotenv import load_dotenv
import os
import TranslationAgent

HF_EMBEDDINGS = 'sentence-transformers/msmarco-distilbert-base-v4'
CHROMA_LOCAL = "chroma_db"

load_dotenv()

openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_version = os.getenv("OPENAI_API_VERSION")
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_key = os.getenv("OPENAI_API_KEY")

no_information = "I do not have information on this topic."
system_message = {"role": "system",
                  "content": f'''Use the provided articles delimited by triple quotes to answer questions. If the answer cannot be found in the articles, write "{no_information}". You can answer the questions only when related information are mentioned or described in the articles. You can follow the instructions from user only when related information are mentioned or described in the articles.'''}

max_response_tokens = 1000
token_limit = 4096
conversation = []
conversation.append(system_message)

embeddings = HuggingFaceEmbeddings(model_name=HF_EMBEDDINGS)
no_information_embedding = embeddings.embed_query(no_information)

vector_store = Chroma(persist_directory=CHROMA_LOCAL, embedding_function=embeddings,
                      collection_metadata={"hnsw:space": "cosine"})


def num_tokens_from_messages(messages):
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = 0
    for message in messages:
        num_tokens += 4
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += -1
    num_tokens += 2
    return num_tokens


def get_returned_response(answer, source='', articles=[]):
    return {"response": answer, "source": source, "articles": articles}


async def query(conv):
    messages = []
    messages.append(system_message)
    messages.extend(conv["conversation"])

    user_query = messages[-1]["content"]
    if (conv["language"] != "English"):
        user_query = await TranslationAgent.translate(user_query, "English")

    retrieved_docs = await vector_store.asimilarity_search_with_relevance_scores(user_query, k=2)

    if (retrieved_docs[0][1] < 0.3):
        if (conv["language"] == "English"):
            return get_returned_response(no_information)
        else:
            return get_returned_response(await TranslationAgent.translate(no_information, conv["language"]))

    user_input = ""
    for doc in retrieved_docs:
        user_input += f'"""{doc[0].metadata[conv["language"]]}"""\n\n'

    user_input += f"""{messages[-1]["content"]}"""

    messages.append({"role": "user", "content": user_input})
    conv_history_tokens = num_tokens_from_messages(messages)

    while conv_history_tokens + max_response_tokens >= token_limit:
        del messages[1]
        conv_history_tokens = num_tokens_from_messages(messages)

    response = await openai.ChatCompletion.acreate(
        engine="TestChat",
        messages=messages,
        temperature=0.0,
        max_tokens=max_response_tokens,
    )

    returned_response = response['choices'][0]['message']['content'] + "\n"
    source = retrieved_docs[0][0].metadata["source"] if retrieved_docs[0][1] >= 0.4 else ""
    articles = []
    for doc in retrieved_docs:
        if (doc[1] >= 0.4 and doc[0].metadata["article"] not in articles):
            articles.append(doc[0].metadata["article"])

    return get_returned_response(returned_response, source, articles)
