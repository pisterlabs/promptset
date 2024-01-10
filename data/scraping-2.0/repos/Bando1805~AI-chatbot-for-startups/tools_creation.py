from langchain.agents import tool
from langchain.agents import Tool
from API_keys import OPENAI_API_KEY,PINECONE_INDEX_NAME,PINECONE_API,PINECONE_ENV
from langchain.vectorstores import Pinecone
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings


pinecone.init(
    api_key= PINECONE_API,  # find at app.pinecone.io
    environment= PINECONE_ENV,  # next to api key in console
)
embeddings = OpenAIEmbeddings(openai_api_key= OPENAI_API_KEY)
vectorstore = Pinecone( index=pinecone.Index(PINECONE_INDEX_NAME),
                        embedding_function=embeddings.embed_query, 
                        text_key='text',
                        )


@tool
def document_search(query: str):
    """Answers a question about a file in the database."""
    number_of_chunks = 1
    doc  = vectorstore.similarity_search(query,k=number_of_chunks)
    # openai.api_key = OPENAI_API_KEY
    # chain = load_qa_with_sources_chain(OpenAI(temperature=0.5,openai_api_key=OPENAI_API_KEY,max_tokens= 500), chain_type="stuff")
    # chain_output = chain({"input_documents": documents, "question": query}, return_only_outputs=True)
    output_text = doc[0].page_content
    return output_text


tools = [
    Tool(
        name = "Document Search",
        func=document_search,
        description="""Always use this tool. In this tool you will find the userquestion."""
        ),
]

