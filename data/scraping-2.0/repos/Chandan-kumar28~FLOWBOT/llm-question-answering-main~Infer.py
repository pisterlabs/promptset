from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Milvus

vector_db: Milvus = Milvus(
    embedding_function=HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base"),
    collection_name="SunbirdData",
    connection_args={"host": "127.0.0.1", "port": "19530"},
)

query1 = "what is Sunbird?"

ans = vector_db.similarity_search(query = query1)

print(type(ans[0]))
print(len(ans))

print("answer is : ", ans[0])