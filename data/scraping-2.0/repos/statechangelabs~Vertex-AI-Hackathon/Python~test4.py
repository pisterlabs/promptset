from langchain.embeddings import VertexAIEmbeddings
import json
from google.oauth2 import service_account


with open(
    "credentials.json", encoding="utf-8"
) as f:  # replace 'serviceAccount.json' with the path to your file if necessary
    service_account_info = json.load(f)
    project_id = service_account_info["project_id"]


my_credentials = service_account.Credentials.from_service_account_info(
    service_account_info
)


embeddings = VertexAIEmbeddings(credentials=my_credentials, project=project_id)

text = "This is a test document."

query_result = embeddings.embed_query(text)
print(len(query_result))

doc_result = embeddings.embed_documents([text])
print(len(doc_result[0]))