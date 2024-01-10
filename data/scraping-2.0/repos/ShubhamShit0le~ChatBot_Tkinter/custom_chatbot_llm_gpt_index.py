
import tkinter as tk
from tkinter import Button, Entry, Scrollbar, Text
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, PromptHelper, StorageContext, load_index_from_storage
from langchain import OpenAI
from llama_index import ServiceContext
import os
import webbrowser
import skyciv

# Set your OpenAI API key here
os.environ["OPENAI_API_KEY"] = 'Your_OpenAI_Key'

z_coord = None
y_coord = None
waiting_for_coordinates = False

def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 256
    # set maximum chunk overlap
    chunk_overlap_ratio = 0.2
    # set chunk size limit
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, chunk_overlap_ratio, chunk_size_limit=chunk_size_limit)

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-002", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

    # Load the index from disk (assuming 'index.json' exists)
    index.storage_context.persist(persist_dir="./content/")

    return index

def ask_bot(user_query):
    storage_context = StorageContext.from_defaults(persist_dir='./content/')
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()
    response = query_engine.query(user_query)
    return response.response

def create_skyciv_model():
    global z_coord
    global y_coord
    global waiting_for_coordinates

    response_text.config(state=tk.NORMAL)  # Allow modifications to the response text

    if waiting_for_coordinates:
        if z_coord is None:
            user_input = entry_user_input.get()
            try:
                z_coord = float(user_input)
                entry_user_input.delete(0, tk.END)  # Clear the input box
                response_text.insert(tk.END, "You: " + user_input + "\n")
                response_text.insert(tk.END, "Bot: Please enter Y-coordinate.\n")
            except ValueError:
                response_text.insert(tk.END, "Bot: Please enter a valid numeric value for Z-coordinate.\n")
        else:
            user_input = entry_user_input.get()
            y_coord = user_input
            entry_user_input.delete(0, tk.END)  # Clear the input box
            response_text.insert(tk.END, "You: " + user_input + "\n")

            if y_coord:
                # Perform SkyCiv operations here with z_coord and y_coord
                model = skyciv.Model("metric")

                model.nodes.add(0, 0, 0)
                model.nodes.add(float(y_coord), 0, 0)
                model.nodes.add(float(y_coord), float(y_coord), 0)
                model.nodes.add(0, float(y_coord), 0)
                model.nodes.add(0, 0, -float(z_coord))
                model.nodes.add(float(y_coord), 0, -float(z_coord))
                model.nodes.add(float(y_coord), float(y_coord), -float(z_coord))
                model.nodes.add(0, float(y_coord), -float(z_coord))

                model.members.add(1, 2, "Continuous", 1, 0, "FFFFFF")
                model.members.add(2, 3, "Continuous", 1, 0, "FFFFFF")
                model.members.add(3, 4, "Continuous", 1, 0, "FFFFFF")
                model.members.add(4, 1, "Continuous", 1, 0, "FFFFFF")
                model.members.add(1, 5, "Continuous", 1, 0, "FFFFFF")
                model.members.add(2, 6, "Continuous", 1, 0, "FFFFFF")
                model.members.add(3, 7, "Continuous", 1, 0, "FFFFFF")
                model.members.add(4, 8, "Continuous", 1, 0, "FFFFFF")

                ao = skyciv.ApiObject()
                ao.auth.username = "Your_User_Name"
                ao.auth.key = "Your_API_Key"
                ao.functions.add("S3D.session.start")
                ao.functions.add("S3D.model.set", {"s3d_model": model})
                ao.functions.add("S3D.file.save", {"name": "package-debut", "path": "api/PIP/"})
                res = ao.request()
                output_url = res["response"]["data"]
                webbrowser.open(output_url)
                response_text.insert(tk.END, "Bot: SkyCiv API Response: " + output_url + "\n")

                # Reset coordinates
                z_coord = None
                y_coord = None
                response_text.insert(tk.END, "Bot: You can enter your message.\n")
                waiting_for_coordinates = False
            else:
                response_text.insert(tk.END, "Bot: Please enter a valid numeric value for Y-coordinate.\n")
    else:
        if entry_user_input.get().lower() == 'skyciv':
            waiting_for_coordinates = True
            entry_user_input.delete(0, tk.END)
            response_text.insert(tk.END, "Bot: Please enter Z-coordinate.\n")
        else:
            user_input = entry_user_input.get()
            response_text.insert(tk.END, "You: " + user_input + "\n")
            entry_user_input.delete(0, tk.END)  # Clear the input box
            bot_response = ask_bot(user_input)  # Call ask_bot with user's query
            response_text.insert(tk.END, "Bot: " + bot_response + "\n")
    
    entry_user_input.focus()
    response_text.config(state=tk.DISABLED) 

# Create a Tkinter window
root = tk.Tk()
root.title("Chatbot")

# Create a Text widget for displaying bot responses
response_text = Text(root, wrap=tk.WORD, width=80, height=30, state=tk.DISABLED)  # Initially read-only
response_text.pack()

# Create a Scrollbar for the Text widget
scrollbar = Scrollbar(root, command=response_text.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
response_text.config(yscrollcommand=scrollbar.set)

# Create an Entry widget for user input
entry_user_input = Entry(root, width=50)
entry_user_input.pack()
entry_user_input.insert(0, "Enter your message")  # Initial prompt for user input

# Create a "Send" button to trigger sending questions or creating SkyCiv models
send_button = Button(root, text="Send", command=create_skyciv_model)
send_button.pack()

# Bind "Enter" key press to create_skyciv_model function
entry_user_input.bind("<Return>", lambda event=None: create_skyciv_model())

# Start the Tkinter main loop
root.mainloop()

# Construct the chatbot index after the Tkinter UI loop starts
index = construct_index("content/")
