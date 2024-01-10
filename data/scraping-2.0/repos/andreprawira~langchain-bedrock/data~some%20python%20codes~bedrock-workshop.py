import warnings
warnings.filterwarnings('ignore')
import os
import sys
from utils import print_ww
import boto3
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from urllib.request import urlretrieve
import numpy as np
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader, DirectoryLoader
from langchain.vectorstores import FAISS
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# module_path = ".."
# sys.path.append(os.path.abspath(module_path))
boto3_bedrock = boto3.client("bedrock-runtime")

# - create the Anthropic Model
llm = Bedrock(model_id="anthropic.claude-v2:1", 
              client=boto3_bedrock, 
              model_kwargs={'max_tokens_to_sample':200})
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=boto3_bedrock)

os.makedirs("data", exist_ok=True)
files = [
    "https://www.irs.gov/pub/irs-pdf/p1544.pdf",
    "https://www.irs.gov/pub/irs-pdf/p15.pdf",
    "https://www.irs.gov/pub/irs-pdf/p1212.pdf",
]
for url in files:
    file_path = os.path.join("data", url.rpartition("/")[2])
    urlretrieve(url, file_path)

#loader = PyPDFDirectoryLoader("./data/")
loader = DirectoryLoader("./data/", glob="*.py")

documents = loader.load()
# - in our testing Character split works better with this PDF data set
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 1000,
    chunk_overlap  = 100,
)
docs = text_splitter.split_documents(documents)

avg_doc_length = lambda documents: sum([len(doc.page_content) for doc in documents])//len(documents)
avg_char_count_pre = avg_doc_length(documents)
avg_char_count_post = avg_doc_length(docs)
# print(f'Average length among {len(documents)} documents loaded is {avg_char_count_pre} characters.')
# print(f'After the split we have {len(docs)} documents more than the original {len(documents)}.')
# print(f'Average length among {len(docs)} documents (after split) is {avg_char_count_post} characters.')

try:
    sample_embedding = np.array(bedrock_embeddings.embed_query(docs[0].page_content))
    # print("Sample embedding of a document chunk: ", sample_embedding)
    # print("Size of the embedding: ", sample_embedding.shape)

except ValueError as error:
    if  "AccessDeniedException" in str(error):
        print(f"\x1b[41m{error}\
        \nTo troubeshoot this issue please refer to the following resources.\
         \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
         \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")      
        class StopExecution(ValueError):
            def _render_traceback_(self):
                pass
        raise StopExecution        
    else:
        raise error
    
vectorstore_faiss = FAISS.from_documents(
    docs,
    bedrock_embeddings,
)

wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=vectorstore_faiss)

prompt_template = """

Human: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, simply say idk lol.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
#query = "will i go to jail if i file my taxes incorrectly?"
query = "apakah user pool region argumen yang diwajibkan? apakah ada nilai bawaan region untuk user pool? jika iya, apakah nilai bawaan untuk user pool region?"
#query = "whos john cena?"

result = qa({"query": query})
print_ww(result['result'])