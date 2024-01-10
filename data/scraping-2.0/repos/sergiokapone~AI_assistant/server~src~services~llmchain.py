from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma

from ..config.settings import settings
from ..vector_db.chroma_init import get_chroma_client
from ..repository.hystory import extract_history

API_KEY = settings.llm_api_key


#llm_id = "databricks/dolly-v2-3b"
llm_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

transformer_id = "sentence-transformers/all-MiniLM-L6-v2"

template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible.
{context}
{chat_history}
Question: {question}
Helpful Answer:"""


prompt = PromptTemplate.from_template(template)

template_no_context = """Answer the question at the end. Use three sentences maximum. Keep the answer as concise as possible.
{chat_history}
Question: {input}
Helpful Answer:"""

prompt_no_context = PromptTemplate.from_template(template_no_context)


class Chain:
    def __init__(self, history=[]):
        self.llm = HuggingFaceHub(
            repo_id=llm_id,
            huggingfacehub_api_token=API_KEY,
            model_kwargs={"temperature": 0.2, "max_length": 255},
        )
        self.chains = {}
        self.embedding_function = SentenceTransformerEmbeddings(model_name=transformer_id)


    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    

    def create_context(self, user_id):
        client = get_chroma_client()
        try:
            client.get_collection(f"collection_{user_id}") #raises ValueError if the collection does't exist
            context = Chroma(
                        client=client,
                        collection_name=f"collection_{user_id}",
                        embedding_function=self.embedding_function,)
            return context
        except ValueError:
            return None
            

    async def create_memory(self, user_id):
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        history = await extract_history(user_id)
        for i in history:
            memory.save_context({"input": i[0]}, {"output": i[1]})

        return memory
        # later this code will add to the memory messages from the database.

    async def create_chain(self, user_id):
        context = self.create_context(user_id)
        if context:
            chain = ConversationalRetrievalChain.from_llm(
                self.llm,
                context.as_retriever(search_kwargs={"k": 3}),
                memory=await self.create_memory(user_id),
            )
        else:
            chain = ConversationChain(
                llm=self.llm,
                memory=await self.create_memory(user_id),
                prompt=prompt_no_context,
            )
        return (chain, context is not None)

    def answer(self, query, user_id):
        if self.chains[user_id][1]:
            result = self.chains[user_id][0]({"question": query})
            return result["answer"].lstrip()
        else:
            return self.chains[user_id][0].predict(input=query).lstrip()
        
    
    async def update(self, user_id):
        self.chains[user_id] = await self.create_chain(user_id)
        

    async def __call__(self, query, user_id):
        if user_id not in self.chains.keys():
            self.chains[user_id] = await self.create_chain(user_id)
        return self.answer(query, user_id)

chain = Chain()