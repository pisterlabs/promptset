import os

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import PythonCodeTextSplitter
from langchain.vectorstores import Chroma
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

with open('Wissenschaftliche_Methoden/all_txt.txt', 'r') as file:
    text = file.read()

# Split text into words
python_splitter = PythonCodeTextSplitter(chunk_size=200, chunk_overlap=0)
docs = python_splitter.create_documents([text])
print(docs)

splitted_text = python_splitter.split_text(text)
print(splitted_text)


embeddings = OpenAIEmbeddings()

store = Chroma.from_documents(docs, embeddings, persist_directory='db')

store.persist()

query = "Wie Schreibt man en Abstract?"
# docs = store.similarity_search(query)



# prompt = st.text_input('Niebo gwazdziste nademna, prawo moralne we mnie. A ty, czym jesteś?Jestem twoją wolnością. Jestem tym, co masz, czego się trzymasz, czym możesz zdecydować i dążyć do tego, co uważasz za słuszne. Jestem tym, co możesz zmienić i wpłynąć na życie innych. Jestem tym, co jest w twojej ręce.')
prompt = 'how ot wirte an abstract?'
search = store.similarity_search_with_score(prompt)
print(search[0][0].page_content)

search.sort(key=lambda x: x[1], reverse=True)

for i in search:
    print("Content:" + i[0].page_content)
    print("Relevance:" + str(i[1]))
    print('------------------')


