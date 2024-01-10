import streamlit as st

st.set_page_config(page_title="Embeddings", page_icon="📔", layout="wide")
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
import os
from myfunc.mojafunkcija import (
    st_style,
    positive_login,
    show_logo,
    pinecone_stats,
    def_chunk,
)
import random
import Pinecone_Utility
import ScrapperH
import PyPDF2
import io
import re
from langchain.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder
import datetime
import json
from uuid import uuid4
from io import StringIO

version = "14.11.23. Hybrid"
st_style()


def main():
    show_logo()
    chunk_size, chunk_overlap = def_chunk()
    #chunk_size = 50
    st.markdown(
        f"<p style='font-size: 10px; color: grey;'>{version}</p>",
        unsafe_allow_html=True,
    )
    st.subheader("Izaberite operaciju za Embeding HYBRID Method")
    with st.expander("Pročitajte uputstvo:"):
        st.caption(
            """
                   Prethodni korak bio je kreiranje pitanja. To smo radili pomoću besplatnog ChatGPT modela. Iz svake oblasti (ili iz dokumenta)
                   zamolimo ChatGPT da kreira relevantna pitanja. Na pitanja mozemo da odgovorimo sami ili se odgovori mogu izvuci iz dokumenta.\n
                   Ukoliko zelite da vam model kreira odgovore, odaberite ulazni fajl sa pitanjma iz prethodnog koraka.
                   Opciono, ako je za odgovore potreban izvor, odaberite i fajl sa izvorom. Unesite sistemsku poruku (opis ponašanja modela)
                   i naziv FT modela. Kliknite na Submit i sačekajte da se obrada završi.
                   Fajl sa odgovorima ćete kasnije korisiti za kreiranje FT modela.\n
                   Pre prelaska na sledeću fazu OBAVEZNO pregledajte izlazni dokument sa odgovorima i korigujte ga po potrebi.
                   """
        )

    if "podeli_button" not in st.session_state:
        st.session_state["podeli_button"] = False
    if "manage_button" not in st.session_state:
        st.session_state["manage_button"] = False
    if "kreiraj_button" not in st.session_state:
        st.session_state["kreiraj_button"] = False
    if "stats_button" not in st.session_state:
        st.session_state["stats_button"] = False
    if "screp_button" not in st.session_state:
        st.session_state["screp_button"] = False

    if "submit_b" not in st.session_state:
        st.session_state["submit_b"] = False

    if "submit_b2" not in st.session_state:
        st.session_state["submit_b2"] = False

    if "nesto" not in st.session_state:
        st.session_state["nesto"] = 0

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        with st.form(key="podeli", clear_on_submit=False):
            st.session_state.podeli_button = st.form_submit_button(
                label="Pripremi Dokument",
                use_container_width=True,
                help="Podela dokumenta na delove za indeksiranje",
            )
            if st.session_state.podeli_button:
                st.session_state.nesto = 1
    with col3:
        with st.form(key="kreiraj", clear_on_submit=False):
            st.session_state.kreiraj_button = st.form_submit_button(
                label="Kreiraj Embeding",
                use_container_width=True,
                help="Kreiranje Pinecone Indeksa",
            )
            if st.session_state.kreiraj_button:
                st.session_state.nesto = 2
    with col4:
        # st.write("Nije dostupno za Hybrid Embeding ")
        with st.form(key="manage", clear_on_submit=False):
            st.session_state.manage_button = st.form_submit_button(
                label="Upravljaj sa Pinecone",
                use_container_width=True,
                help="Manipulacije sa Pinecone Indeksom",
            )
            if st.session_state.manage_button:
                st.session_state.nesto = 3
    with col5:
        with st.form(key="stats", clear_on_submit=False):
            index = pinecone.Index("embedings1")
            st.session_state.stats_button = st.form_submit_button(
                label="Pokaži Statistiku",
                use_container_width=True,
                help="Statistika Pinecone Indeksa",
            )
            if st.session_state.stats_button:
                st.session_state.nesto = 4
    with col2:
        # st.write("Nije dostupno za Hybrid Embeding ")
        with st.form(key="screp", clear_on_submit=False):
            st.session_state.screp_button = st.form_submit_button(
                label="Pripremi Websajt", use_container_width=True, help="Scrape URL"
            )
            if st.session_state.screp_button:
                st.session_state.nesto = 5
    st.divider()
    phmain = st.empty()

    if st.session_state.nesto == 1:
        with phmain.container():
            prepare_embeddings(chunk_size, chunk_overlap)
    elif st.session_state.nesto == 2:
        with phmain.container():
            do_embeddings()
    elif st.session_state.nesto == 3:
        with phmain.container():
            Pinecone_Utility.main()
    elif st.session_state.nesto == 4:
        with phmain.container():
            index = pinecone.Index("positive")
            api_key = os.getenv("PINECONE_API_KEY_POS")
            env = os.getenv("PINECONE_ENVIRONMENT_POS")
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            index_name = "positive"
            pinecone.init(api_key=api_key, environment=env)
            index = pinecone.Index(index_name)
            pinecone_stats(index, index_name)
    elif st.session_state.nesto == 5:
        with phmain.container():
            ScrapperH.main(chunk_size, chunk_overlap)


def prepare_embeddings(chunk_size, chunk_overlap):
    skinuto = False
    napisano = False

    file_name = "chunks.json"
    with st.form(key="my_form_prepare", clear_on_submit=False):
        st.subheader("Učitajte dokumenta i metadata za Pinecone Indeks")

        dokum = st.file_uploader(
            "Izaberite dokument/e", key="upload_file", type=["txt", "pdf", "docx"]
        )
        # define delimiter
        text_delimiter = st.text_input(
            "Unesite delimiter: ",
            help="Delimiter se koristi za podelu dokumenta na delove za indeksiranje. Prazno za paragraf",
        )
        # define prefix
        text_prefix = st.text_input(
            "Unesite prefiks za tekst: ",
            help="Prefiks se dodaje na početak teksta pre podela na delove za indeksiranje",
        )
        add_schema = st.radio(
            "Da li želite da dodate Schema Data (može značajno produžiti vreme potrebno za kreiranje): ",
            ("Da", "Ne"),
            key="add_schema_doc",
            help="Schema Data se dodaje na početak teksta",
        )
        st.session_state.submit_b = st.form_submit_button(
            label="Submit",
            help="Pokreće podelu dokumenta na delove za indeksiranje",
        )
        st.info(f"Chunk veličina: {chunk_size}, chunk preklapanje: {chunk_overlap}")
        if len(text_prefix) > 0:
            text_prefix = text_prefix + " "

        if dokum is not None and st.session_state.submit_b == True:
            with io.open(dokum.name, "wb") as file:
                file.write(dokum.getbuffer())

            if text_delimiter == "":
                text_delimiter = "\n\n"

            if ".pdf" in dokum.name:
                pdf_reader = PyPDF2.PdfReader(dokum)
                num_pages = len(pdf_reader.pages)
                text_content = ""

                for page in range(num_pages):
                    page_obj = pdf_reader.pages[page]
                    text_content += page_obj.extract_text()
                text_content = text_content.replace("•", "")
                text_content = re.sub(r"(?<=\b\w) (?=\w\b)", "", text_content)
                with io.open("temp.txt", "w", encoding="utf-8") as f:
                    f.write(text_content)

                loader = UnstructuredFileLoader("temp.txt", encoding="utf-8")
            else:
                # Creating a file loader object
                loader = UnstructuredFileLoader(dokum.name, encoding="utf-8")

            data = loader.load()

            # Split the document into smaller parts, the separator should be the word "Chapter"
            text_splitter = CharacterTextSplitter(
                separator=text_delimiter,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            texts = text_splitter.split_documents(data)


            # # Create the OpenAI embeddings
            st.write(f"Učitano {len(texts)} tekstova")

            # Define a custom method to convert Document to a JSON-serializable format
            output_json_list = []
            # Loop through the Document objects and convert them to JSON
            # Define your lists
            names = ["Miljan", "Goran", "Darko", "Nemanja"]
            topics = ["RAG indeksi", "AI asistenti", "Epic", "Positive doo"]

            i = 0
            for document in texts:
                i += 1
                
                # Randomly assign a name and a topic
                person_name = random.choice(names)
                topic = random.choice(topics)

                output_dict = {
                    "id": str(uuid4()),
                    "chunk": i,
                    "text": text_prefix + document.page_content,
                    "source": document.metadata.get("source", ""),
                    "date": datetime.datetime.now().strftime("%d.%m.%Y"),
                    "person_name": person_name,
                    "topic": topic,
                }
                output_json_list.append(output_dict)

            # # Specify the file name where you want to save the JSON data
            json_string = (
                "["
                + ",\n".join(
                    json.dumps(d, ensure_ascii=False) for d in output_json_list
                )
                + "]"
            )

            # Now, json_string contains the JSON data as a string

            napisano = st.info(
                "Tekstovi su sačuvani u JSON obliku, downloadujte ih na svoj računar"
            )

    if napisano:
        file_name = os.path.splitext(dokum.name)[0]
        skinuto = st.download_button(
            "Download JSON",
            data=json_string,
            file_name=f"{file_name}.json",
            mime="application/json",
        )
    if skinuto:
        st.success(f"Tekstovi sačuvani na {file_name} su sada spremni za Embeding")


def do_embeddings():
    with st.form(key="my_form_do", clear_on_submit=False):
        err_log = ""
        # Read the texts from the .txt file
        chunks = []
        dokum = st.file_uploader(
            "Izaberite dokument/e",
            key="upload_json_file",
            type=[".json"],
            help="Izaberite dokument koji ste podelili na delove za indeksiranje",
        )

        # Now, you can use stored_texts as your texts
        # with st.form(key="my_form2", clear_on_submit=False):
        namespace = st.text_input(
            "Unesi naziv namespace-a: ",
            help="Naziv namespace-a je obavezan za kreiranje Pinecone Indeksa",
        )
        submit_b2 = st.form_submit_button(
            label="Submit", help="Pokreće kreiranje Pinecone Indeksa"
        )
        if submit_b2 and dokum and namespace:
            stringio = StringIO(dokum.getvalue().decode("utf-8"))

            # Directly load the JSON data from file content
            data = json.load(stringio)

            # Initialize lists outside the loop
            my_list = []
            my_meta = []

            # Process each JSON object in the data
            for item in data:
                # Append the text to my_list
                my_list.append(item['text'])
    
                # Append other data to my_meta
                meta_data = {key: value for key, value in item.items() if key != 'text'}
                my_meta.append(meta_data)
                
            # Initialize OpenAI and Pinecone API key
            api_key = os.getenv("PINECONE_API_KEY_POS")
            env = os.getenv("PINECONE_ENVIRONMENT_POS")
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            index_name = "positive"

            pinecone.init(api_key=api_key, environment=env)
            index = pinecone.Index(index_name)
            embeddings = OpenAIEmbeddings()

            # upsert data
            bm25_encoder = BM25Encoder()
            # fit tf-idf values on your corpus
            bm25_encoder.fit(my_list)

            retriever = PineconeHybridSearchRetriever(
                embeddings=embeddings,
                sparse_encoder=bm25_encoder,
                index=index,
            )

            retriever.add_texts(texts=my_list, metadatas=my_meta, namespace=namespace)

            # gives stats about index
            st.info("Napunjen Pinecone")

            st.success(f"Sačuvano u Pinecone-u")
            pinecone_stats(index, index_name)


# Koristi se samo za deploy na streamlit.io
deployment_environment = os.environ.get("DEPLOYMENT_ENVIRONMENT")

if deployment_environment == "Streamlit":
    name, authentication_status, username = positive_login(main, " ")
else:
    if __name__ == "__main__":
        main()
