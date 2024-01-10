from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.document_transformers.embeddings_redundant_filter import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors.base import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain.prompts import PromptTemplate
from libs.using_ffm import get_embed, get_ffm

embeddings_zh = get_embed()


retrievers = [FAISS.load_local("/home/ubuntu/AFS_tools/use_cases/webcrawler_chatbot/generated/all_docs_embedding_website",
                               embeddings_zh).as_retriever( 
                                                           search_type="mmr", 
                                                           search_kwargs={"k": 3, "include_metadata": True}),
             ]

print(f"retriever loaded {len(retrievers)} embeddings")

lotr = MergerRetriever(retrievers=retrievers)
filter = EmbeddingsRedundantFilter(embeddings=embeddings_zh)
pipeline = DocumentCompressorPipeline(transformers=[filter])
compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline, 
    base_retriever=lotr
)




ffm = get_ffm()

prompt_template = """
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{{
{context}
}}


'{question}'
 
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}

qa_endpoint = RetrievalQA.from_chain_type(llm=ffm,
                                          chain_type='stuff',
                                          retriever=lotr, 
                                          chain_type_kwargs=chain_type_kwargs,
                                          return_source_documents=True
                                          )



class QUERY(BaseModel):
    q: str | None = None

class API_RESULT(BaseModel):
    answer: str | None = None

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/ask/")
async def read_item(query: QUERY):
    qa_ans = qa_endpoint(query.q)
    ans = API_RESULT()
    ans.answer = qa_ans['result']
    return JSONResponse(jsonable_encoder(ans))



if __name__ == "__main__":
    pass



