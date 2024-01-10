# Please make sure to copy the code to a local environment and add the Secret API Key provided in the separate document.

#Import of the necessary libraries and packages

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
from openai import OpenAI
import os
from PyPDF2 import PdfReader
import json
import pandas as pd
import time

# Initialize the OpenAI  client and prompt the API key to initialize the OpenAI client

client = OpenAI(
    api_key=input("Enter the openai api key: ")  # Enter the API key
)

# Function to extract text from a PDF file (turn it into a string)
def extract_text_from_pdf(pdf_file_path):
    # Initialize an empty string to hold the extracted text
    text_content = ""
    # Open the PDF file in read-binary mode
    with open(pdf_file_path, 'rb') as file:
        pdf_reader = PdfReader(file) # Create a PDF reader object
        # Iterate through each page in the PDF file
        for page in pdf_reader.pages:
            # Append the extracted text of each page to the text_content string
            text_content += page.extract_text() + "\n"
    return text_content

# Function to send a user message to an OpenAI thread
def send_message(thread_id, user_message):
    # Call the OpenAI API to create a new message in the specified thread
    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_message
    )
    # Return the response message object
    return message

# Function to check whether the analysis by the api is completed 
def wait_on_run(run_id, thread_id):
    # Enter a loop that continues until the run is complete or fails
    while True:
        # Retrieve the status of the run using its ID and the thread ID
        run = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_id,
        )
        # Check if the run status is 'completed' and return the run object if so
        if run.status == "completed":
            return run
        # If the run failed or was cancelled, raise an exception
        elif run.status in ["failed", "cancelled"]:
            raise Exception(f"Run failed or was cancelled: {run.status}")
        # Pause the loop for 1 second before checking the status again
        time.sleep(1)

# Function to run the assistant and retrieve the response
def run_assistant_and_get_response(assistant_id, thread_id, last_response_id=None):
    # Create a new run with the specified assistant ID and thread ID
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id
    )
    # Wait for the run to complete and then retrieve the list of messages in the thread
    run = wait_on_run(run.id, thread_id)

    messages = client.beta.threads.messages.list(
        thread_id=thread_id
    )
    # Initialize an empty list to hold the answers
    answers = []
    latest_response_id = last_response_id
    # Iterate through each message in the thread
    for message in messages.data:
        # Check if the message role is 'assistant' and it's a new message since the last response ID
        if message.role == "assistant" and (last_response_id is None or message.id > last_response_id):
            try:
                 # Extract the text value of the message content and append to the answers list
                answer = message.content[0].text.value
                answers.append(answer)
            except AttributeError:
                # If there is no text value, print "No reply"
                print("No reply")
            # Update the latest response ID for the next iteration
            latest_response_id = message.id
    # Return the latest response ID and the list of answers
    return latest_response_id, answers

# Function to display DataFrame in a Treeview widget
def display_dataframe_in_treeview(df):
    # Create a new Toplevel window
    top_level = tk.Toplevel(window)
    top_level.title("DataFrame Output")
    
    # Create the Treeview widget with the correct column identifiers
    columns = list(df.columns)
    tree = ttk.Treeview(top_level, columns=columns, show='headings')
    
    # Generate the headings
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, anchor="w")
    
    # Insert the data into the Treeview
    for index, row in df.iterrows():
        # Ensure that the values are passed in the same order as the columns
        tree.insert('', 'end', values=row[columns].tolist())
    
    # Add vertical scrollbar to the Treeview
    scrollbar_vertical = ttk.Scrollbar(top_level, orient='vertical', command=tree.yview)
    tree.configure(yscrollcommand=scrollbar_vertical.set) # Link scrollbar to the Treeview
    scrollbar_vertical.pack(side='right', fill='y') # Pack scrollbar to the UI

    #Add horizontal scrollbar to the Treeview
    scrollbar_horizontal = ttk.Scrollbar(top_level, orient='horizontal', command=tree.xview)
    tree.configure(xscrollcommand=scrollbar_horizontal.set) # Link scrollbar to the Treeview
    scrollbar_horizontal.pack(side='bottom', fill='x') # Pack scrollbar to the UI

    tree.pack(expand=True, fill='both') # Pack Treeview to the UI to occupy the available space

# Function to process the results of the poem analysis
def process_analysis_results(analysis_result):
    # Parse the JSON string into a Python dictionary
    analysis_output = json.loads(analysis_result)
    # Check if the expected key 'poemAnalysisOutput' is in the JSON
    if 'poemAnalysisOutput' in analysis_output:
        # Retrieve the poem analysis data
        poem_data = analysis_output['poemAnalysisOutput']
        # If 'analysis' is a nested dictionary, we normalize it first 
        if 'analysis' in poem_data:
            # Flatten the nested 'analysis' data into a flat structure
            analysis_flat = pd.json_normalize(poem_data['analysis'])
            # Update the 'poem_data' with the flattened analysis data
            poem_data.update(analysis_flat.to_dict(orient='records')[0])
            # Remove the now redundant nested 'analysis' key
            del poem_data['analysis']
        # Create a DataFrame from the poem analysis data
        df = pd.DataFrame([poem_data])  # The data is in a dictionary, so let's make a list out of it
        display_dataframe_in_treeview(df)
    else:
        # Inform the user if no analysis data was found in the result
        messagebox.showinfo("Result", "No analysis found in the result.")

# Function allowing to select a pdf file
def select_pdf_file():
    # Open a file dialog to select a PDF file
    file_path = filedialog.askopenfilename(
        title="Select a PDF file",
        filetypes=[("PDF files", "*.pdf")] # Restrict file dialog to only show PDF files
    )
    # Restrict file dialog to only show PDF files
    if file_path:
        entry_pdf_path.delete(0, tk.END) # Clear any existing content in the entry
        entry_pdf_path.insert(0, file_path) # Insert the selected file path

# Function to process the text for analysis
def process_text(text):
    try:
        # ID of the assistant to use for analysis
        existing_assistant_id = "asst_E3qfm6X0yQam3oNuHPy7Zq79"
        # Create a new thread to communicate with the assistant
        thread = client.beta.threads.create()
        # Send the text to the assistant for processing
        send_message(thread.id, text)
        # Wait for the response from the assistant and get answers
        last_response_id, answers = run_assistant_and_get_response(existing_assistant_id, thread.id)

        # If answers were received, process them
        if answers:
            # Debug: Print the answer to check if it's valid JSON
            print("Received answer:", answers[0])
            try:
                # Process the analysis results and display them
                process_analysis_results(answers[0])
            except json.JSONDecodeError as e:
                # Handle JSON parsing errors
                messagebox.showerror("Error", f"An error occurred in JSON parsing: {e}")
        else:
            # Inform the user if no answers were received
            messagebox.showinfo("Result", "No answers were received for analysis.")
    except Exception as e:
        # Handle other exceptions and display an error message
        messagebox.showerror("Error", f"An error occurred: {e}")



# GUI Functions

# Function to handle the input choice (PDF or Text)
def on_input_choice():
    # If the user chooses PDF, hide the text input and show PDF input options
    if input_choice.get() == 'PDF':
        text_input.pack_forget()
        entry_pdf_path.pack(padx=10, pady=5)
        button_select_pdf.pack(pady=5)
        button_analyze.pack(pady=5)
    elif input_choice.get() == 'Text':
        # If the user chooses Text, hide the PDF input and show text input options
        entry_pdf_path.pack_forget()
        button_select_pdf.pack_forget()
        text_input.pack(padx=10, pady=5)
        button_analyze.pack(pady=5)

# Function to handle PDF analysis
def analyze_pdf():
    # Retrieve the file path from the entry field
    pdf_file_path = entry_pdf_path.get()
    # Check if the file exists
    if not os.path.isfile(pdf_file_path):
        messagebox.showerror("Error", "The specified file was not found.")
        return
    
    # Extract text from the PDF and process it
    pdf_text = extract_text_from_pdf(pdf_file_path)
    process_text(pdf_text)

# Function to get the text of poem for the analysis
def analyze_text():
    # Retrieve the text from the scrolledtext widget
    user_text = text_input.get('1.0', tk.END).strip()
    # Check if the text is not empty
    if not user_text:
        messagebox.showerror("Error", "No text to analyze.")
        return
    # Process the text
    process_text(user_text)

# GUI Setup

# Create the main window
window = tk.Tk()
window.title("Poem Analysis Tool")

# Variable to store the input choice
input_choice = tk.StringVar(value='PDF')

# Radio buttons for input choice (PDF or text)
radio_pdf = tk.Radiobutton(window, text="Upload PDF", variable=input_choice, value='PDF', command=on_input_choice)
radio_text = tk.Radiobutton(window, text="Enter Text", variable=input_choice, value='Text', command=on_input_choice)
radio_pdf.pack(anchor='w', padx=10, pady=5)
radio_text.pack(anchor='w', padx=10, pady=5)

# PDF path entry
entry_pdf_path = tk.Entry(window, width=50)

# Select PDF button
button_select_pdf = tk.Button(window, text="Select PDF", command=select_pdf_file)

# Text input area for direct text entry
text_input = scrolledtext.ScrolledText(window, height=10)

# Analyze button for both PDF and text input
button_analyze = tk.Button(window, text="Analyze", command=lambda: analyze_pdf() if input_choice.get() == 'PDF' else analyze_text())

# Initial input choice setup
on_input_choice()

# Start the Tkinter main loop to display and run the GUI
window.mainloop()