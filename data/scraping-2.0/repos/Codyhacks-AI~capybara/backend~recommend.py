from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
# from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader

# from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import pinecone

PINECONE_API_KEY = "a6e82686-204b-40f0-8220-a5c4d5f7d149"
PINECONE_API_ENV = "us-west1-gcp-free"

def get_suggestions(new_code):
    # initialize pinecone database
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
    index_name = "test"

    embeddings = OpenAIEmbeddings()

    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    llm = OpenAI(temperature=0.1)

    prompt_template = """You are an expert coder. Some current code includes the below:
        {context}
        
        You just wrote this new code: {new_code}
        Give suggestions on improvements for the new code.
        Give me line numbers and comments. Send no other text. Only respond with the line and comment. 
        """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "new_code"])

    chain = LLMChain(llm=llm, prompt=PROMPT)

    # generate context via semantic search -> store in docs
    def answer(new_code):
        docs = docsearch.similarity_search(new_code)
        combined = ""
        for doc in docs:
            combined += doc.page_content + "/n"
        print(combined)
        inputs = [{"context": combined, "new_code": new_code}]
        return chain.apply(inputs)

    return answer(new_code)
