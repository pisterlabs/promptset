import openai
import pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
import IPython
from tqdm import tqdm

with open("/Users/michael/Desktop/wip/openai_credentials.txt", "r") as f:
    OPENAI_API_KEY = f.readline().strip()
    openai.api_key = OPENAI_API_KEY

with open("/Users/michael/Desktop/wip/pinecone_credentials.txt", "r") as f:
    PINECONE_API_KEY = f.readline().strip()
    PINECONE_API_ENV = f.readline().strip()
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)


def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]


loader = TextLoader("../context_data/test_articles_clean.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="\n")
texts = text_splitter.split_documents(documents)

index_name = "buffetbot"
pinecone_service = pinecone.Index(index_name=index_name)

texts = texts[1000:]

for idx, text in tqdm(enumerate(texts), total=len(texts)):
    try:
        embeddings = get_embedding(text.page_content)
        vector = {
            "id": str(idx),
            "values": embeddings,
            "metadata": {
                "category": "news",
                "original_text": text.page_content,
            },
        }
        upsert_response = pinecone_service.upsert(
            vectors=[vector],
            namespace="data",
        )
    except Exception as e:
        print(e)
