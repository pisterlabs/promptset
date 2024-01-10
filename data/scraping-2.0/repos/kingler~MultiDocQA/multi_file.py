import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.router import MultiRetrievalQAChain
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader


# Directory where the .txt files are stored
docs_dir = 'docs'

# Initialize the OpenAI embeddings
embedding = OpenAIEmbeddings()

retrievers = []
retriever_descriptions = []
retriever_names = []

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap  = 20,
    length_function = len,
)

# Iterate over all .txt and .pdf files in the directory
for filename in os.listdir(docs_dir):
    doc = None
    # Check if a persistent Chroma VectorStore already exists for this document
    if os.path.exists(filename[:-4]):
        # If it exists, load it from disk
        retriever = Chroma(persist_directory=filename[:-4], embedding_function=embedding).as_retriever()
    else:
        # Load the document and split it
        if filename.endswith('.txt'):
            try:
                with open(os.path.join(docs_dir, filename), 'r', encoding='utf-8') as f:
                    doc = f.read()
            except UnicodeDecodeError:
                # Handle possible encoding errors
                print(f"Skipping file {filename} due to encoding errors.")
                continue
            # If it's a .txt, we split the document
            doc = text_splitter.create_documents([doc])
        elif filename.endswith('.pdf'):
            loader = PyPDFLoader(os.path.join(docs_dir, filename))
            doc = loader.load_and_split()
            print(doc)
        
        if doc is not None:
            # Create a new Chroma VectorStore and save it to disk
            retriever = Chroma.from_documents(documents=doc, embedding=embedding, persist_directory=filename[:-4])
            retriever.persist()
            retriever = retriever.as_retriever()
        
    # Add the retriever, its name and its description to the respective lists
    retrievers.append(retriever)
    retriever_names.append(filename[:-4])
    # PAY ATTENTON TO THE NAMES OF THE FILES AS THEY WILL BE IN THE DESCRIPTIONS
    retriever_descriptions.append(f"Good for answering questions about {filename[:-4]}")


# Initialize the MultiRetrievalQAChain
chain = MultiRetrievalQAChain.from_retrievers(OpenAI(), retriever_names, retriever_descriptions, retrievers, verbose=True)

# Test it
# print(chain.run("What are the differences between Newton and Feynman?"))

while True:
    print(chain.run(input("What would you like to know?>>> ")))

