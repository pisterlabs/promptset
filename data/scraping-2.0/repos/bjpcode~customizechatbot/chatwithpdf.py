import os
import pandas as pd
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain

os.environ["OPENAI_API_KEY"] = ""

"""# 1. Loading PDFs and chunking with LangChain"""

# You MUST add your PDF to local files in this notebook (folder icon on left hand side of screen)

# Simple method - Split by pages
loader = PyPDFLoader("C:/Users/baij/Desktop/新媒体平台.pdf")
pages = loader.load_and_split()
#print(pages[0])

# SKIP TO STEP 2 IF YOU'RE USING THIS METHOD
chunks = pages

# Advanced method - Split by chunk

# Step 1: Convert PDF to text
#import textract
#doc = textract.process("C:/Users/baij/Desktop/新媒体平台.pdf")

# Step 2: Save to .txt and reopen (helps prevent issues)
#with open('C:/Users/baij/Desktop/新媒体平台.txt', 'w') as f:
#    f.write(doc.decode('utf-8'))

#with open('新媒体平台.txt', 'r') as f:
#    text = f.read()
def read_txt_file(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

# Example usage:
text = read_txt_file("C:/Users/baij/Desktop/testfile.txt")
#print(text)

# Step 3: Create function to count tokens
#tokenizer = GPT2TokenizerFast.from_pretrained("Xenova/gpt-3.5-turbo")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Step 4: Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 512,
    chunk_overlap  = 24,
    length_function = count_tokens,
)

chunks = text_splitter.create_documents([text])

# Result is many LangChain 'Documents' around 500 tokens or less (Recursive splitter sometimes allows more tokens to retain context)
type(chunks[0])

# Quick data visualization to ensure chunking was successful

# Create a list of token counts
token_counts = [count_tokens(chunk.page_content) for chunk in chunks]

# Create a DataFrame from the token counts
df = pd.DataFrame({'Token Count': token_counts})

# Create a histogram of the token count distribution
df.hist(bins=40, )

# Show the plot
#plt.show()

"""# 2. Embed text and store embeddings"""

# Get embedding model
embeddings = OpenAIEmbeddings()

# Create vector database
db = FAISS.from_documents(chunks, embeddings)

"""# 3. Setup retrieval function"""

# Check similarity search is working
query = "Who created transformers?"
docs = db.similarity_search(query)
docs[0]

# Create QA chain to integrate similarity search with user queries (answer query from knowledge base)

chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

query = "Who created transformers?"
docs = db.similarity_search(query)

chain.run(input_documents=docs, question=query)

"""# 5. Create chatbot with chat memory (OPTIONAL)"""

from IPython.display import display
import ipywidgets as widgets

# Create conversation chain that uses our vectordb as retriver, this also allows for chat history management
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), db.as_retriever())

chat_history = []

def on_submit(_):
    query = input_box.value
    input_box.value = ""

    if query.lower() == 'exit':
        print("Thank you for using the State of the Union chatbot!")
        return

    result = qa({"question": query, "chat_history": chat_history})
    chat_history.append((query, result['answer']))

    display(widgets.HTML(f'<b>User:</b> {query}'))
    display(widgets.HTML(f'<b><font color="blue">Chatbot:</font></b> {result["answer"]}'))

#print("Welcome to the Transformers chatbot! Type 'exit' to stop.")

#input_box = widgets.Text(placeholder='Please enter your question:')
#input_box.on_submit(on_submit)

#display(input_box)
def chat_with_bot():
    print("Welcome to bjp chatbot! Type 'exit' to stop.")
    while True:
        query = input("Please enter your question: ")

        if query.lower() == 'exit':
            print("Thank you for using bjp chatbot!")
            break

        result = qa({"question": query, "chat_history": chat_history})
        chat_history.append((query, result['answer']))

        print(f"Chatbot: {result['answer']}")

chat_with_bot()
