import os, yaml, uvicorn
from fastapi import FastAPI
from langserve import add_routes
from langchain.vectorstores import FAISS
from langchain.schema import StrOutputParser
from fastapi.responses import RedirectResponse
from langchain.llms import HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import FastEmbedEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

app = FastAPI()

with open('/Users/1zuu/Desktop/LLM RESEARCH/LLMPro/cadentials.yaml') as f:
    credentials = yaml.load(f, Loader=yaml.FullLoader)

os.environ['HUGGINGFACEHUB_API_TOKEN'] = credentials['HUGGINGFACEHUB_API_TOKEN']
os.environ['AZURE_OPENAI_API_VERSION'] = credentials['AD_OPENAI_API_VERSION']
os.environ['AZURE_OPENAI_ENDPOINT'] = credentials['AD_OPENAI_API_BASE']
os.environ['AZURE_OPENAI_API_KEY'] = credentials['AD_OPENAI_API_KEY']

# difine LLMs
# hf_llm = HuggingFaceEndpoint(
#                         endpoint_url="https://oolbderhhrn6klkc.us-east-1.aws.endpoints.huggingface.cloud",
#                         huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
#                         task="text-generation",
#                         )

hf_llm = AzureChatOpenAI(
                        openai_api_version=credentials['AD_OPENAI_API_VERSION'],
                        azure_deployment=credentials['AD_DEPLOYMENT_ID'],
                        )
embeddings_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# loadf the index
faiss_index = FAISS.load_local("langserve_faiss", embeddings_model)
retriever = faiss_index.as_retriever()

# define the prompt
prompt_template = """\
Use the provided context to answer the user's question. If you don't know the answer, say you don't know.

Context:
{context}

Question:
{question}"""

rag_prompt = ChatPromptTemplate.from_template(prompt_template)

entry_point_chain = RunnableParallel({
                                    "context": retriever, 
                                    "question": RunnablePassthrough()
                                    })
rag_chain = entry_point_chain | rag_prompt | hf_llm | StrOutputParser()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

add_routes(app, rag_chain, path="/rag")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)