import os 
from dotenv import load_dotenv # to load keys
from langchain.document_loaders.csv_loader import CSVLoader #to load csv file
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings # to embedd the csv file
from langchain.vectorstores import FAISS # to store the vectore in the database
from langchain.prompts import PromptTemplate #for proper prompting
from langchain.llms import GooglePalm # model
from langchain.chains import RetrievalQA  # using the model and retrievar to produce result

#load the keys
load_dotenv()

#hugging face api key
hug_api_key = os.getenv('HUG_API_KEY')

#google api key
api_key = os.getenv('API_KEY')

# Initialise LLM with required params
model = GooglePalm(google_api_key=api_key,temperature=0, max_tokens=1000)
# Initialise Embedding with required params
embedding_method = HuggingFaceInferenceAPIEmbeddings(
    api_key=hug_api_key,
    model_name="sentence-transformers/all-MiniLM-l6-v2"
)

# function to load csv
def load_csv(path):
    #load the csv file
    loader = CSVLoader(file_path=path)

    # Store the loaded data in the 'data' variable
    data = loader.load()
    return data

# function for embedding
def vector_convertion(docs):
    vectorindex = FAISS.from_documents(docs, embedding_method)
    return vectorindex

# function for chain
def get_chain(vectorindex):

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""


    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}



    chain = RetrievalQA.from_chain_type(llm=model,
                                chain_type="stuff",
                                retriever=vectorindex.as_retriever(score_threshold = 0.7),
                                input_key="query",
                                return_source_documents=True,
                                chain_type_kwargs=chain_type_kwargs)
    return chain

if __name__ == "__main__":
    data = load_csv("input.csv")
    vector_index = vector_convertion(data)
    chain = get_chain(vector_index)
    response = chain("How much revenue did Potato Inc. make from Japan in Q4 2022?")
    print(response['result'])
    