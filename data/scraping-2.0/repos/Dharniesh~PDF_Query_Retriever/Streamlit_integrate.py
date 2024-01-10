import os
import pinecone
import tempfile
import PyPDF2
from langchain.vectorstores import Pinecone
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import streamlit as st

# Check to see if there is an environment variable with your API keys, if not, use what you put below
OPENAI_API_KEY = st.secrets["openai_api"]
PINECONE_API_KEY = st.secrets["pinecone_api"]
PINECONE_API_ENV = st.secrets["pinecone_env"]


api_key = PINECONE_API_KEY

def query_find(query):
    print('inside queryfind')
    # Perform the similarity search and get the documents
    docs = docsearch.similarity_search(query=query, k=1)

    # Assuming 'docs' is the list containing your documents or elements
    pg_ref = [doc.metadata['page'] + 1 for doc in docs]

    result = chain.run(input_documents=docs, question=query)

    if pg_ref:
        page_number = pg_ref[0]
        return result, page_number
    else:
        print('Nothing to print')


def upload_text_batch(page_texts, metadata):
    print('upload text batch')
    #index.upsert(page_texts)
    vectorstore.add_texts(page_texts, metadata,batch_size=100) 

def process_pdf(pdf_path):
    print('process pdf')
    # Load the merged PDF file
    loader = PyPDFLoader(pdf_path)
    data = loader.load()

    # Split the PDF into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=0)
    texts = text_splitter.split_documents(data)

    # Upload the contents and metadata to Pinecone
    metadata = [t.metadata for t in texts]
    page_texts = [t.page_content for t in texts]

    return page_texts, metadata


# Function to perform the document search and return the results
def perform_search(pdf_paths):
    print('perform search')
    # Check if the file paths exist
    if not all(os.path.exists(path) for path in pdf_paths):
        st.warning("One or more PDF files do not exist.")
        return 'Nothing to Print 1', 'Nothing to Print 2'

    # Initialize the vectorstore once for all batch uploads
    
    #vectorstore = Pinecone(index, embeddings.embed_query, lambda i: metadata[i])

    page_texts_list = []
    metadata_list = []
    for pdf_path in pdf_paths:
        page_texts, metadata = process_pdf(pdf_path)
        page_texts_list.extend(page_texts)
        metadata_list.extend(metadata)

    upload_text_batch(page_texts_list,metadata_list)


# Streamlit app
def main():
    st.title("PDF Document Search and Question Answering")
    print('init and checking pinecone')
    pinecone.init(api_key=api_key, environment=PINECONE_API_ENV)
    if not pinecone.list_indexes():
        pinecone.create_index(dimension=1536, name='intelpdf', metric='cosine')
    # File Upload
    uploaded_files = st.file_uploader("Choose multiple PDF files to merge and search:", type=["pdf"],
                                      accept_multiple_files=True)

    if uploaded_files is not None and len(uploaded_files) >= 1:
        print('inside upload files main')
        # Create a temporary directory to store the uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_paths = []
            for i, file in enumerate(uploaded_files):
                # Save the uploaded PDF files to the temporary directory
                temp_path = os.path.join(temp_dir, f"uploaded_pdf_{i}.pdf")
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                pdf_paths.append(temp_path)

            if st.button("Done", key="done_button"):
                st.text("Uploading files...")
                perform_search(pdf_paths)
                st.session_state.perform_search_done = True  # Set the session state variable to indicate files are uploaded

    # Wait for files to be uploaded and perform_search to be done before showing the query input
    if "perform_search_done" in st.session_state and st.session_state.perform_search_done:
        query = st.text_input("Enter your question:")

        if query:
            con, pgn = query_find(query)
            if con is not None and pgn is not None:
                st.subheader("Answer:")
                st.write(con)
                st.subheader("Page Number:")
                st.write(pgn)
        else:
            st.warning("Please enter a question.")

print('embeddings')
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
print('llm')
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
print('docsearch')
docsearch = Pinecone.from_existing_index(index_name='intelpdf', embedding=embeddings)
print('index')
index = pinecone.Index("intelpdf",pool_threads=30)
print('vectorstore')
vectorstore = Pinecone(index, embeddings.embed_query, 'text')
print('chain')
chain = load_qa_chain(llm, chain_type="stuff")

if __name__ == "__main__":
    main()
