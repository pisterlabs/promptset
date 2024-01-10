from llama_index.llms import OpenAI
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
import os
from llama_index.embeddings import HuggingFaceEmbedding

class RAGPaLMQuery:
    def __init__(self):
        os.environ['OPENAI_API_KEY'] = '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'

        documents = SimpleDirectoryReader("./competition").load_data()


        embed_model = HuggingFaceEmbedding(model_name='BAAI/bge-large-en-v1.5')

        llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")


        service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, chunk_size=800, chunk_overlap=20)

        index = VectorStoreIndex.from_documents(documents, service_context=service_context, show_progress=True)

        # Create a query engine from the index
        query_engine = index.as_query_engine(similarity_top_k=10)
        response = query_engine.query(
            "I'm a student interested in getting a loan. Can you help me?"
        )
        print(response)