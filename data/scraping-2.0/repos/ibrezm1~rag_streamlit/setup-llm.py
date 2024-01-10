import vertexai
import os
from langchain.llms import VertexAI

PROJECT_ID = 'zeta-yen-319702'
REGION = 'us-central1'
BUCKET = 'gs://zeta-yen-319702-temp/'

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './svc-gcp-key.py'

vertexai.init(
    project=PROJECT_ID,
    location=REGION,
    staging_bucket=BUCKET
)

# https://cloud.google.com/vertex-ai/docs/generative-ai/learn/models#:~:text=PaLM%202%20for%20Text%20(%20text%2Dunicorn%20),with%20complex%20natural%20language%20tasks.


llm = VertexAI(
    #model_name="text-bison@001",
    model_name="text-unicorn",
    max_output_tokens=256,
    temperature=0.8,
    top_p=0.8,
    top_k=5,
    verbose=False,
)

print(llm(prompt = "hi"))

from langchain.embeddings import TensorflowHubEmbeddings
from langchain.vectorstores import DeepLake
url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
embed_model = TensorflowHubEmbeddings(model_url=url)

db = DeepLake(dataset_path='./deeplk', embedding=embed_model)


from langchain.chains import ConversationalRetrievalChain
qa = ConversationalRetrievalChain.from_llm(
    llm=VertexAI(temperature=0.5,top_p=0.8,top_k=0.8),
    retriever=db.as_retriever(),
    return_source_documents=True,verbose=True
)


query = "What was the Mad Hatter's riddle about raven and writing desks?"
result = qa({"question": query,"chat_history" : ""})

print(result)