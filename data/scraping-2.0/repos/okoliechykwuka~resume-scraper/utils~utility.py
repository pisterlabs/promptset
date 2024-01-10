from llama_index import ServiceContext, set_global_service_context
from llama_index.llms import OpenAI
from llama_index.embeddings import resolve_embed_model
from llama_index.indices.postprocessor import SentenceTransformerRerank
# from dotenv import load_dotenv, find_dotenv

# _ = load_dotenv(find_dotenv()) # read local .env file
embed_model = resolve_embed_model("local:BAAI/bge-large-en-v1.5")

llama_llm = OpenAI(model="gpt-3.5-turbo-16k-0613")

service_context = ServiceContext.from_defaults(
    llm=llama_llm, embed_model=embed_model
)
set_global_service_context(service_context)

rerank = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3
        )