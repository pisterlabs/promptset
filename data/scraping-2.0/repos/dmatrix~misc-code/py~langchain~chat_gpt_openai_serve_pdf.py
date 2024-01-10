import os
import requests
from pathlib import Path

from langchain.document_loaders import PyPDFLoader 
from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from ray import serve
from starlette.requests import Request

# simple example inspired from 
# https://medium.com/geekculture/automating-pdf-interaction-with-langchain-and-chatgpt-e723337f26a6
# and https://github.com/sophiamyang/tutorials-LangChain/tree/main
# Default model used is OpenAI: default-gpt-3.5-turbo

@serve.deployment(route_prefix="/",
                  autoscaling_config={
                        "min_replicas": 2,
                        "initial_replicas": 2,
                        "max_replicas": 4,
    })
class AnswerPDFQuestions():
    def __init__(self, vector_db_path: str, open_ai_key: str):

        # Load the PDF and create the db for embeddings if one does not exsist
        # Or read from existing index 
        os.environ["OPENAI_API_KEY"] = open_ai_key
        embeddings = OpenAIEmbeddings()
        self._vectordb = Chroma(persist_directory=vector_db_path, embedding_function=embeddings)
        self._pdf_qa_chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.9), 
                                                             self._vectordb.as_retriever())
        self._chat_history=[]

    async def __call__(self, http_request: Request) -> str:
        json_request: str = await http_request.json()
        prompts = []
        for prompt in json_request:
            text = prompt["text"]
            if isinstance(text, list):
                prompts.extend(text)
            else:
                prompts.append(text)
        result = self._pdf_qa_chain({"question": prompts[0], "chat_history": self._chat_history})
        
        return result
    
if __name__ == "__main__":

    KEY = "your_key"
    vector_db = Path(Path.cwd(), "vector_oai_db").as_posix()
    deployment = AnswerPDFQuestions.bind(vector_db, KEY)
    serve.run(deployment)

    # send the request to the deployment
    prompts = [ "What is the total number of publications?",
                # "What is the percentage increase in the number of AI-related job postings?",
                # "What are the top takeaways from this report?",
                # "List benchmarks are released to evaulate AI workloads?",
                # "Describe and summarize the techincal ethical issues raised in the report?",
                # "Why Chinese citizens are more optimistic about AI than Americans?",
                "How many bills containing “artificial intelligence” were passed into law?",
                # "What is the percentage increase in the number of AI-related job postings?"
    ]
    
    sample_inputs = [{"text": prompt} for prompt in prompts]
    for sample_input in sample_inputs:
        output = requests.post("http://localhost:8000/", json=[sample_input]).json()
        print(f"Question: {output['question']}\nAnswer: {output['answer']}\n")
        
    serve.shutdown()
