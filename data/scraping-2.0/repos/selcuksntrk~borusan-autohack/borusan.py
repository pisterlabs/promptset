import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
from llama_index.llms import OpenAI

def main():
    # Initialize session state for the index and query engine
    if 'index' not in st.session_state or 'query_engine' not in st.session_state:
        documents = SimpleDirectoryReader(
            input_files=["Main_Data.pdf"]
        ).load_data()

        document = Document(text="\n\n".join([doc.text for doc in documents]))

        llm = OpenAI(model="gpt-3.5-turbo", temperature=0.2)
        service_context = ServiceContext.from_defaults(
            llm=llm, embed_model="local:BAAI/bge-large-en-v1.5"
        )
        st.session_state['index'] = VectorStoreIndex.from_documents([document], service_context=service_context)
        st.session_state['query_engine'] = st.session_state['index'].as_query_engine(similarity_top_k=10)

    st.title("Borusan AutoInsight")
    user_query = st.text_input("Merak Ettiğiniz Model Hakkındaki Sorunuzu Giriniz:")
    submit_button = st.button('Sor')  # Add this line for the button

    if submit_button and user_query:
        response = st.session_state['query_engine'].query(user_query)
        st.text("Yanıt:")
        st.write(str(response))

if __name__ == "__main__":
   main()
