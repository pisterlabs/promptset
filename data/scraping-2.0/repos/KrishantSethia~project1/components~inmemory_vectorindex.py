from llama_index.llms import OpenAI
from llama_index import StorageContext, ServiceContext, VectorStoreIndex
import os
os.getenv("OPENAI_API_KEY")


class InMemoryVectorIndex():
    def create_vector_index(nodes):
        llm = OpenAI(model="gpt-4")
        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(nodes)
        service_context = ServiceContext.from_defaults(chunk_size=256, llm=llm)
        index = VectorStoreIndex(
            storage_context=storage_context,
            service_context=service_context,
            nodes=nodes
        )

        return index
