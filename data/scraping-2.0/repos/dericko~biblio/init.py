import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv
load_dotenv()


from PyPDF2 import PdfReader

reader = PdfReader("text.pdf")
res = []
meta = []
ids = []
i = 1
length = len(reader.pages)
for page in reader.pages:
    res.append(page.extract_text())
    meta.append(
        {
            "loc": i / length,
            "title": "Safari",
            "author": "Jennifer Egan",
            "genre": "short story",
        }
    )
    ids.append(str(i))
    i += 1


client = chromadb.Client()
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"], model_name="text-embedding-ada-002"
)
collection = client.create_collection("all-books", embedding_function=openai_ef)
collection.add(documents=res, metadatas=meta, ids=ids)
query_result = collection.query(query_texts=["What was the narrator's name?"], n_results=3)

top_page = query_result['documents'][0][0];

# TODO: query
import openai
openai.api_key = os.environ["OPENAI_API_KEY"]
res = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "I'm asking questions about the short story Safari by Jennifer Egan. Please keep your answers short and simple."},
        {"role": "user", "content": "Here is a relevant page from the story:"},
        {"role": "user", "content": top_page},
        {"role": "user", "content": "What was the narrator's name?"},
    ]
)
print(res)
