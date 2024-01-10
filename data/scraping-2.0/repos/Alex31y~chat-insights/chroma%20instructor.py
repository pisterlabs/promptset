import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from langchain.document_loaders import PyPDFLoader
#from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.utilities import WikipediaAPIWrapper
import re
import openai
import os
import threading
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from InstructorEmbedding import INSTRUCTOR

client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="chromadb" # Optional, defaults to .chromadb/ in the current directory
))
#emb_fn = embedding_functions.DefaultEmbeddingFunction()
emb_fn = embedding_functions.InstructorEmbeddingFunction(model_name="hkunlp/instructor-large", device="cuda")
chroma_client = chromadb.Client()

collection = client.get_or_create_collection(name="docs", embedding_function=emb_fn)

chunk_size = 500
n_chunks = 15

def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text

def to_text(file, url):
    text = ""
    if url:
        wikipedia = WikipediaAPIWrapper()
        pages = wikipedia.load_file(file_paths)
        #print(pages)
        page = preprocess(pages)
        text += page
    elif file:
        print(file)
        file_extension = os.path.splitext(file)[1]
        # Check if the file extension is ".txt" or ".word"
        if file_extension == ".txt":
            print("File extension is .txt")
            loader = UnstructuredFileLoader(file)
        elif file_extension == ".pdf":
            print("File extension is .pdf")
            loader = PyPDFLoader(file)
        else:
            print("File extension not supported")
            return

        pages = loader.load_and_split()
        for page in pages:
            page = page.page_content
            page = preprocess(page)
            text += page
    else:
        text = "input non riuscito"
    return text

def text_to_chunks(text, overlap_percentage = 0.2):
    chunks = []
    overlap_size = int(chunk_size * overlap_percentage)
    start = 0
    end = chunk_size
    counter = 1

    while start < len(text):
        chunk = text[start:end]
        chunk_with_counter = f'[{counter}] {chunk}'
        chunks.append(chunk_with_counter + '\n\n')
        start += chunk_size - overlap_size
        end = start + chunk_size
        counter += 1
    corpus = []
    for chunk in chunks:
        piezzo = ['Represent the document for retrieval: ', chunk]
        corpus.append(piezzo)
    return corpus

import pickle
def registra_file(string, filename):
    try:
        # Load existing data from the file
        data = get_file_list(filename)
        if data is None:
            data = []  # Create an empty list if the file doesn't exist or is empty

        # Append the new string to the data list
        data.append(string)

        # Save the updated data back to the file
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
    except Exception as e:
        print(f"Error saving file '{filename}': {str(e)}")

def get_file_list(filename):
    try:
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error loading file '{filename}': {str(e)}")
        return None

class SemanticSearch:
    def fit(self, data, file):
        filename = os.path.basename(file)
        print("Numero di chunks nel documento: ")
        print(len(data))
        id_list = [f"{filename}_{index}" for index in range(1, len(data)+1)]
        meta_data = [{"file": filename} for index in range(1, len(data)+1)]
        collection.add(documents=data, metadatas=meta_data, ids=id_list)
        registra_file(filename, 'files.pkl')

    # restituisco i top n chunks più simili alla domanda
    def __call__(self, text): # text è la domanda input dell'utente
        domanda = [['Represent question for retrieving supporting documents: ', text]]
        results = collection.query(
            query_texts=domanda,
            n_results=n_chunks,
            include=["documents"]
        )
        print(type(results['documents']))

        return results['documents']

def generate_text(openAI_key, prompt, engine="text-davinci-003"):
    openai.api_key = openAI_key
    completions = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=500,     #incide sulla lunghezza della risposta output, quindi sul prezzo della chiamata
        n=1,
        stop=None,
        temperature=0.7,    #più è basso meno si spende, forse
    )
    message = completions.choices[0].text
    return message

def generate_answer(question, openAI_key):
    # genero i chunks
    topn_chunks = recommender(question) # metodo __call__ : confronto l'embedding della domanda all'embedding del pdf ed ottengo gli n snippet di testo più vicini
    prompt = ""
    prompt += 'search results:\n\n'
    for c in topn_chunks:
        prompt += ' '.join(c)

    prompt += "Instructions: Compose a comprehensive reply to the query using the search results given. " \
              "If the search results mention multiple subjects with the same name, create separate answers for each. " \
              "Only include information found in the results and don't add any additional information. " \
              "Make sure the answer is correct and don't output false content. " \
              "If the text does not relate to the query, simply state 'Non è stata trovata una risposta alla tua domanda nel testo'." \
              "Ignore outlier search results which has nothing to do with the question. Only answer what is asked. " \
              "The answer should be short and concise. Answer step-by-step. \n\nQuery: {question}\nAnswer: "

    prompt += f"Query: {question}\nAnswer:"

    """
    file_object = open('docs\domande.txt', 'a')
    file_object.write('\n\n\nprompt:\n')
    file_object.write(prompt)
    # Close the file
    file_object.close()
    """

    #answer = generate_text(openAI_key, prompt, "text-davinci-003")
    answer = prompt
    print(prompt)
    return answer

def ask_question():
    query = query_entry.get()
    api_key = key_entry.get()
    answer = generate_answer(query, api_key)
    text_area.pack()
    text_area.delete(1.0, tk.END)
    text_area.insert(tk.END, answer)

def clean_collection():
    file_list = get_file_list('files.pkl')
    for file in file_list:
        collection.delete(where={"file": file})
        print(f"rimossi chunk di: {file}")
    if os.path.exists("files.pkl"):
        # Delete the file
        os.remove("files.pkl")
        print("rimosso il picke")
    create_scrollable_list(window, "")

def load_file():
    while not stop_event.is_set():
        progress_bar.pack()
        progress_bar.start()
        url = url_entry.get()
        if(url):
            print("url true")
            text = to_text(url, True)
            chunks = text_to_chunks(text)
            recommender.fit(chunks)
        else:
            print("url false")
            global file_paths
            for file in file_paths:
                text = to_text(file, False)
                chunks = text_to_chunks(text)
                recommender.fit(chunks, file)

        # Create the scrollable list
        file_list = get_file_list('files.pkl')
        create_scrollable_list(window, file_list)
        progress_bar.stop()
        progress_bar.pack_forget()
        stop_event.set()  # Set the stop event to stop the thread
        break

#chiama load_file
def start_thread():
    global thread, stop_event
    if thread and thread.is_alive():
        print("già in esecuzione")
        return  # Do nothing if a thread is already running
    stop_event.clear()  # Reset the stop event

    # Start the new thread
    stop_event = threading.Event()
    thread = threading.Thread(target=load_file)
    thread.start()

def input_file():
    global file_paths
    filetypes = [("PDF Files", "*.pdf"), ("Text Files", "*.txt"), ("Word Files", "*.docx")]
    file_paths = filedialog.askopenfilenames()

def create_scrollable_list(root, elements):
    if not elements:
        return  # If elements is empty, do nothing

    # Create a frame to hold the listbox and scrollbar
    frame = tk.Frame(root)
    frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Create a scrollbar
    scrollbar = tk.Scrollbar(frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Create a listbox and associate it with the scrollbar
    listbox = tk.Listbox(frame, yscrollcommand=scrollbar.set)
    listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Configure the scrollbar to control the listbox
    scrollbar.config(command=listbox.yview)

    # Add elements to the listbox
    for element in elements:
        listbox.insert(tk.END, element)


def update_sizes(event=None):
    window.update_idletasks()  # Update the window to get the current size
    text_area.configure(width=(window.winfo_width() // 10), height=(window.winfo_height() // 25))


recommender = SemanticSearch()
# Create the main window
window = tk.Tk()
window.title("Chat Insights")
window.geometry("600x400")

file_paths = None
select_button = tk.Button(window, text="Select Files", command=input_file)
select_button.pack()
# Create button to retrieve the query and API key
upload_button = tk.Button(window, text="Upload", command=start_thread)
upload_button.pack()
delete_button = tk.Button(window, text="Flush", command=clean_collection)
delete_button.pack()
stop_event = threading.Event()
thread = None
progress_bar = ttk.Progressbar(window, mode='indeterminate')
progress_bar.pack_forget()
url_label = tk.Label(window, text="inserisci una pagina Wikipedia:")
url_label.pack()
url_entry = tk.Entry(window, width=60)
url_entry.pack()
key_label = tk.Label(window, text="API Key:")
key_label.pack()
key_entry = tk.Entry(window, width=60)
key_entry.pack()
query_label = tk.Label(window, text="Fai la tua domanda:")
query_label.pack()
query_entry = tk.Entry(window, width=80)
query_entry.pack()



submit_button = tk.Button(window, text="Chiedi", command=ask_question)
submit_button.pack()
# Create a text area to display the extracted text
text_area = tk.Text(window)
progress_bar.pack_forget()

# Make the input boxes and text area adjust dynamically
window.bind('<Configure>', update_sizes)
query_entry.pack_propagate(False)
text_area.pack_propagate(False)

# Create the scrollable list
file_list = get_file_list('files.pkl')
create_scrollable_list(window, file_list)

# Start the main event loop
window.mainloop()
