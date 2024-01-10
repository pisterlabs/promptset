import os
import streamlit as st
import bibtexparser  # Import bibtexparser

from embedding import create_embedding
from global_var import chunk, overlap

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import PythonCodeTextSplitter
from langchain.vectorstores import Chroma

# Function to get bibtex citations
def get_bibtex_citations(bibtex_file):
    with open(bibtex_file, 'r') as f:
        bib_database = bibtexparser.load(f)
    return {entry['ID']: entry for entry in bib_database.entries}

def open_pdf_at_page(pdf_path, page_number):
    """Open a PDF at a specific page using evince."""
    os.system(f"evince --page-label={page_number} {pdf_path}.pdf")

# Load bibtex citations
PAPERS_DIR = "./papers"
bibtex_citations = get_bibtex_citations(os.path.join(PAPERS_DIR, 'citations.bib'))

# Split text into words
python_splitter = PythonCodeTextSplitter(chunk_size=chunk, chunk_overlap=overlap)
embeddings = create_embedding()

db = Chroma(persist_directory='db', embedding_function=embeddings)

st.title('🦜🔗 Semantic Search with OpenAI and Streamlit')

# Create a text input box for the user
prompt = st.text_input('Enter your search query')

if prompt:
    search = db.similarity_search_with_score(prompt, k=5)
    search.sort(key=lambda x: x[1], reverse=True)
    for index, i in enumerate(search):
        st.write("Paper Name: " + i[0].metadata['paper_name'])
        st.write("Page Number: " + str(i[0].metadata['page_num']))
        st.write("Content:" + i[0].page_content)
        st.write("Relevance:" + str(i[1]))

        # Form a unique button key by adding index
        unique_button_key = f"{i[0].metadata['paper_name']}_{i[0].metadata['page_num']}_{index}"

        if st.button(f'Open {i[0].metadata["paper_name"]} at page {i[0].metadata["page_num"]}', key=unique_button_key):
            open_pdf_at_page(os.path.join(PAPERS_DIR, i[0].metadata['paper_name']), i[0].metadata['page_num'])
        
        # Get the bibtex citation
        bibtex_entry = bibtex_citations.get(i[0].metadata['paper_name'], None)
        print(bibtex_entry)  # Debugging-Ausgabe
        if bibtex_entry:
            from bibtexparser.bibdatabase import BibDatabase  # Import here if you want or at the top of your file

            bib_database = BibDatabase()
            bib_database.entries = [bibtex_entry]
            bibtex_text = bibtexparser.dumps(bib_database)
            st.text_area(f"BibTeX Citation for {i[0].metadata['paper_name']}", bibtex_text, height=200, key=f"bibtex_{unique_button_key}")

        st.write('------------------')
