import os
import openai
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# Define the path to your transcript files and persistence directory
transcript_dir = 'transcripts'
persist_directory = 'db'
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Function to read each transcript file
def read_transcript(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function to split transcripts into chunks
def split_transcripts(texts):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="\n")
    return [text_splitter.split_text(text) for text in texts]

# Function to initialize or load the Chroma database
def initialize_chroma_db(texts):
    if os.path.exists(persist_directory):
        # Load existing database
        print("Loading existing database...")
        db = Chroma(persist_directory="./db", embedding_function=embedding_function)
    else:
        # Create and initialize new database
        #embeddings = OpenAIEmbeddings()
        print("Creating new database...")
        db = Chroma.from_texts(texts, embedding_function, persist_directory=persist_directory)
    return db.as_retriever()

# Main function to process and query transcripts
def main():
    openai.api_key = os.environ["OPENAI_API_KEY"]
    flattened_texts = []
    
    #check if db exists
    if os.path.exists(persist_directory):
       #don't process transcripts
         print("Database exists, skipping transcript processing...")
    else: 
        # Process transcripts
        texts = []
        for filename in os.listdir(transcript_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(transcript_dir, filename)
                transcript_content = read_transcript(file_path)
                texts.append(transcript_content)

        split_texts = split_transcripts(texts)
        flattened_texts = [item for sublist in split_texts for item in sublist]
    
    print("Initializing database...")
    docsearch = initialize_chroma_db(flattened_texts)

    # Example query and processing
    query = "Based on all of the transcripts, summarize who is Aaron LeBauer and what does he know about physical therapy?"
    docs = docsearch.get_relevant_documents(query)
    chat_model = ChatOpenAI(model_name="gpt-4-1106-preview")
    chain = load_qa_chain(llm=chat_model, chain_type="stuff")
    answer = chain.run(input_documents=docs, question=query)

    print("Answer:", answer)

if __name__ == "__main__":
    main()
