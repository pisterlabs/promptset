import os
from PyPDF2 import PdfReader
import pickle

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

############ TEXT LOADERS ############
# Functions to read different file types
def read_pdf(file_path):
    with open(file_path, "rb") as file:
        pdf_reader = PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

def read_word(file_path):
    doc = docx.Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def read_txt(file_path):
    with open(file_path, "r") as file:
        text = file.read()
    return text

def read_documents_from_directory(directory):
    combined_text = ""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith(".pdf"):
            combined_text += read_pdf(file_path)
        elif filename.endswith(".docx"):
            combined_text += read_word(file_path)
        elif filename.endswith(".txt"):
            combined_text += read_txt(file_path)
    return combined_text

###############################################

train_directory = 'train_files/'
merged_data_file = 'merged_data.pkl'

# Check if merged data file exists
if os.path.exists(merged_data_file):
    # Load the merged data from the file
    with open(merged_data_file, 'rb') as file:
        text = pickle.load(file)
else:
    # Read and merge the data from the directory
    text = read_documents_from_directory(train_directory)
    
    # Save the merged data to a file
    with open(merged_data_file, 'wb') as file:
        pickle.dump(text, file)

# split into chunks
char_text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, 
                                      chunk_overlap=200, length_function=len)

text_chunks = char_text_splitter.split_text(text)
  
# create embeddings
openai_api_key = "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"  # Replace with your actual OpenAI key
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
docsearch = FAISS.from_texts(text_chunks, embeddings)

llm = OpenAI(openai_api_key=openai_api_key)
chain = load_qa_chain(llm, chain_type="stuff")

##################################################

# Ask a question
query = "I went to an event where there were people debating violently about the use of guns. I tried talking to them but i was assaulted. As my lawyer, help me file a lawsuit. what are the steps i need to take? let's think step by step"
lawyer_query = "Pretend that you are a lawyer, please provide your professional perspective on the following matter: " + query

temperature = 1
# max_tokens = 200
model = 'gpt-4'

docs = docsearch.similarity_search(lawyer_query)
 
response = chain.run(input_documents=docs, question=lawyer_query)

print(lawyer_query)
print(" ")
print(response)
  
# # If you want to keep track of your spending
# with get_openai_callback(openai_api_key=openai_api_key) as cb:
#     response = chain.run(input_documents=docs, question=query) 
#     print(cb)