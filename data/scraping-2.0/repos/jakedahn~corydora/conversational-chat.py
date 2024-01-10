from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import PromptLayerCallbackHandler

from memory import ConversationWithSourcesBufferMemory

memory = ConversationWithSourcesBufferMemory(
    memory_key="chat_history", return_messages=True
)

embedding = OpenAIEmbeddings()

# Now we can load the persisted database from disk, and use it as normal.
vectordb = Chroma(
    persist_directory="./chroma.db",
    collection_name="aquarium-co-op-youtube",
    embedding_function=embedding,
)

promptlayer_callback = PromptLayerCallbackHandler(pl_tags=["langchain"])

qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(
        model_name="gpt-3.5-turbo-16k",
        temperature=0,
        callbacks=[promptlayer_callback],
    ),
    vectordb.as_retriever(search_kwargs={"k": 25}),
    memory=memory,
    return_source_documents=True,
)

chat_history = []


from datetime import datetime


def get_related_videos(source_documents, limit=5):
    video_data = []
    for doc in source_documents:
        metadata = doc.metadata
        video_data.append(
            {
                "url": f"{metadata['url']}?t={metadata['start']}",
                "title": metadata["title"],
                "thumbnail": metadata["thumbnail"],
                "publishedAt": datetime.fromisoformat(
                    metadata["publishedAt"].rstrip("Z")
                ),
            }
        )
    return video_data[:limit]


question = "What is the best food for cherry neocardina shrimp?"
resp = qa({"question": question, "chat_history": chat_history})

chat_history.append((question, resp["answer"]))

print("question: ", question)
print("answer: ", resp["answer"])
print("source_docs: ", get_related_videos(resp["source_documents"]))
chat_history.append((question, resp["answer"]))


question2 = "Great, what are the best water parameters for cherry neocardina shrimp?"
resp2 = qa({"question": question2, "chat_history": chat_history})

print("question2: ", question2)
print("answer2: ", resp2["answer"])
# print("source_docs2: ", resp["source_documents"])

import ipdb

ipdb.set_trace()
