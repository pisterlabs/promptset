import os
import pinecone

from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.tools import Tool
from langchain.vectorstores import Pinecone
from langchain.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

embeddings_model = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.environ["HUGGINGFACEHUB_API_TOKEN"],
    api_url=os.environ["EMBEDDED_ENDPOINT"],
)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

index_name = "demo-index"

index = pinecone.Index(index_name)
docsearch = Pinecone(index, embeddings_model, "text")

docSearchTemplate = """
You are an intelligent, supportive and honest assistant. Don't provide a wrong answer if you are unsure of the context. Answer the question to the best of your ability.
Begin!
Question: {question}
Answer:"""

llm = HuggingFaceEndpoint(
    endpoint_url=os.environ["LLM_ENDPOINT"],
    task="text2text-generation",
    model_kwargs={"max_new_tokens": 1000},
)

prompt = PromptTemplate(template=docSearchTemplate, input_variables=["question"])
docsearchChain = LLMChain(prompt=prompt, llm=llm)

pdf_tool = Tool(
    name="pdf retrieval tool",
    func=docsearchChain.run,
    retriever=docsearch,
    description="Used to retrieve Java Fullstack and SRE curricula information from the pdfs",
    retriever_top_k=3,
)
