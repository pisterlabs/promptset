import os
import streamlit as st
from global_var import chunk, overlap
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import PythonCodeTextSplitter
from langchain.vectorstores import Chroma

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Split text into words
python_splitter = PythonCodeTextSplitter(chunk_size=chunk, chunk_overlap=overlap)
embeddings = OpenAIEmbeddings()
# embeddings = HuggingFaceEmbeddings("sentence-transformers/all-mpnet-base-v2")
embeddings = HuggingFaceEmbeddings()
db = Chroma(persist_directory='db', embedding_function=embeddings)

query = "Wie Schreibt man en Abstract?"
# docs = store.similarity_search(query)



# prompt = st.text_input('Niebo gwazdziste nademna, prawo moralne we mnie. A ty, czym jesteś?Jestem twoją wolnością. Jestem tym, co masz, czego się trzymasz, czym możesz zdecydować i dążyć do tego, co uważasz za słuszne. Jestem tym, co możesz zmienić i wpłynąć na życie innych. Jestem tym, co jest w twojej ręce.')
prompt = st.text_input('Niebo Moralne we mnie , niebo gwiazdziste nade mna , a ty czym jestes?')
search = db.similarity_search_with_score(prompt)
search.sort(key=lambda x: x[1], reverse=True)




st.title('🦜🔗 Semantic Search with OpenAI and Streamlit')
# Create a text input box for the user
if prompt:
    # Then pass the prompt to the LLM
    # response = agent_executor.run(prompt)
    # ...and write it out to the screen
    # st.write(response)

    for i in search:
        st.write("Content:" + i[0].page_content)
        st.write("Relevance:" + str(i[1]))
        st.write('------------------')

    # # With a streamlit expander
    # with st.expander('Document Similarity Search'):
    #     # Find the relevant pages
    #     search = store.similarity_search_with_score(prompt)
    #     # Write out the first
    #     st.write(search[0][0].page_content)
