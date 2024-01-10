import os
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
from dotenv import load_dotenv
load_dotenv()

# textSplitter = RecursiveCharacterTextSplitter(chunkSize=1500, chunkOverlap=100)
# embedder = OpenAIEmbeddings()
pinecone.init(
    environment=os.environ.get("PINECONE_ENVIRONMENT"),
    api_key=os.environ.get("PINECONE_API_KEY")
)

index = pinecone.Index('gpt-test-pdf')
print(index.describe_index_stats())
# async def main():
#     # read article
#     with open('allText.txt', 'r') as file:
#         article = file.read()
#         splittedText = await textSplitter.createDocuments([article])
#         PineconeStore.fromDocuments(splittedText, embedder,pineconeIndex=index,
#         namespace='namespace1'
#     )

#main()