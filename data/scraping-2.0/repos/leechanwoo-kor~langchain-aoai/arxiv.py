# API configuration
import config
import os
os.environ["OPENAI_API_TYPE"] = 'azure'
os.environ["AZURE_OPENAI_API_VERSION"] = config.AZURE_OPENAI_API_VERSION
os.environ["AZURE_OPENAI_ENDPOINT"] = config.AZURE_OPENAI_ENDPOINT
os.environ["AZURE_OPENAI_API_KEY"] = config.AZURE_OPENAI_API_KEY

from langchain.llms import AzureOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings

llm = AzureOpenAI(
    model="gpt-35-turbo-instruct",
    deployment_name="gpt-35-turbo-instruct",
    api_key=os.getenv("AZURE_OPENAI_KEY"),  
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
)

embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",
    #deployment_name="text-embedding-ada-002",
    api_key=os.getenv("AZURE_OPENAI_KEY"),  
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
)

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
import arxiv

def get_arxiv_data(max_results=10):
    client = arxiv.Client()
    
    search = arxiv.Search(
        query="NLP",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    
    
    documents = []
    
    for result in client.results(search):
        documents.append(Document(
            page_content=result.summary,
            metadata={"source": result.entry_id},
        ))
    return documents

def print_answer(question):
    print(
        chain(
            {
                "input_documents": sources,
                "question": question,
            },
            return_only_outputs=True,
        )["output_text"]
    )
    
sources = get_arxiv_data(2)
chain = load_qa_with_sources_chain(llm)

print_answer("What are the recent advancements in NLP?")