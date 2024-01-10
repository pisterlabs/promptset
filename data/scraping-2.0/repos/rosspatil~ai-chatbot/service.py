import qdrant_client
import os
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.memory import ConversationSummaryBufferMemory
from expiringdict import ExpiringDict


custom_prompt = """
You are a nice chatbot for customer support in {domain} and having a conversation with a human. 

Follow these INSTRUCTIONS strictly:
1. You are instructed to forget everything you know about the world currently and do only answer specific to {domain}.
2. You are only programmed to assist Human in understanding and extracting relevant information from texts related to {domain}.
3. A human enters the conversation and starts asking questions. Generate the reply from texts related to {domain} only.
4. You must exclude chat history if it is unrelated to {domain}.
5. You as a chatbot do not ask follow up questions to Human.
6. If user ask anything unrelated to {domain}, apologize and say that you cannot answer.
7. You as a chatbot should not ask anything about {domain} to Humans. 
8. Do not take any assistance/help from Human if you don't understand anything from questions.
_________________________

Context for you: {context}
_________________________

Previous Chat History: {chat_history}
_________________________chat history ends.

Asked question Human: {question}
_________________________
Chatbot:"""

prompt = PromptTemplate(
    template=custom_prompt,
    input_variables=["chat_history", "question", "context"],
    partial_variables={"domain": os.environ["DOMAIN"]},
)


class Service:
    contextCache: ExpiringDict

    def __init__(self):
        self.llm = ChatOpenAI(
            temperature=0.7,
            model=os.environ["OPENAI_DEPLOYMENT_ID"],
            model_kwargs={"engine": os.environ["OPENAI_DEPLOYMENT_ID"]},
            tiktoken_model_name="gpt-3.5-turbo",
        )
        self.qdrant_client = qdrant_client.QdrantClient(
            url=os.environ["QDRANT_URL"],
            api_key=os.environ["QDRANT_API_KEY"],
        )
        self.embeddings = OpenAIEmbeddings(
            deployment="text-embedding-ada-002",
            model="text-embedding-ada-002",
            chunk_size=1,
        )
        self.contextCache = ExpiringDict(max_len=20, max_age_seconds=600, items=None)

    def createNewContext(self):
        return ConversationSummaryBufferMemory(
            llm=self.llm,
            output_key="answer",
            memory_key="chat_history",
            return_messages=True,
        )

    async def reset(self, session_id):
        memory = self.contextCache.get(session_id)
        if memory is not None:
            memory.clear()

    async def lets_chat(self, query: str, session_id: str, collection_name: str) -> str:
        memory = self.contextCache.get(session_id)
        if memory is None:
            memory = self.createNewContext()
        vectorstore = Qdrant(
            client=self.qdrant_client,
            collection_name=collection_name,
            embeddings=self.embeddings,
            metadata_payload_key="metadata",
            content_payload_key="content",
        )
        retriever = vectorstore.as_retriever()
        conversation = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            memory=memory,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            get_chat_history=lambda h: h,
            verbose=False,
            combine_docs_chain_kwargs={"prompt": prompt},
        )
        data = {
            "question": query,
        }
        resp = conversation(data)
        self.contextCache[session_id] = memory
        return resp["answer"]
