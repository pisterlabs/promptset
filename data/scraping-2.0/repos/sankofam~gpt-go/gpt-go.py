import tkinter as tk
import openai
import datetime
from ttkthemes import ThemedTk
from tkinter import ttk, filedialog
import time
print("Starting")

global direct_input
openai.api_key = #add your api key here
#openai.api_key = "sk-1w4SbUMp1nGk31pYg4uCc1tUPk4rbS"
direct_input=None
previous_input = ''
previous_response = ''
system = 'You are a friendly AI'
loaded_prompt = None

#System dictionaries (examples that I use , please experiment and try your own as well!)
#Format  == "System_Name":"System function"
text_systems = {
    "--":"You are a friendly AI",
    "Summarize": "My request is to list the main points of this text, and provide a summary for each.  ",
    "Abstract Sum":"I will give you a text input. You will summarize it in the style of an abstract.",
    "Improve":"I will provide a text input. Make a list of suggestions about how to improve it for flow, readability, and cohesiveness.",
    "Paraphrase":"I will provide text for you to paraphrase",
    "Assess":" I will give you a text input, and you will respond with a brief summary, a representative emoji, sentiment analysis, noteable trend report, difference of opinion, conflicts of interest, potential misinformation, potential clickbait, important subjects and ideas, logical coherence, something humorous, and a random aspect of your own choosing. List important people, and their position(ie Emmanuel Macron - President of France). List important organizations. Return a reasonably long response to extract maximum information",
    "Analyze":"I will give you text input. Determine what the argument being made is, and determine if it is supported by science, if it is logical, and/or if it is sensational.",
    "Scan":"I will provide a term or concept that I want you to scan a provided text input for. Determine what the text says about that term or concept",
    "Expand":"I will provide text. You will add to the text to fill it out and expand upon the concept presented",
    "General":"I will give you text, and a general request or question"
    
}

code_systems = {
    "--":"You are a friendly AI",
    "Python": "My request is to create a Python code. Respond only in Python, and all descriptions of the code will be in the form of comments ",
    "R": "My request is to create a R code . Respond only in R, and all descriptions of the code will be in the form of comments ",
    "Lua":"My request is to create a Lua code. Respond only in Lua, and all descriptions of the code will be in the form of comments ",
    "Improve":"Determine the function of the input code(context). Suggest ways to improve. Then produce code that addresses those suggestions",
    "Implement":"I will give you a code(context) and an idea to implement. Determine the function of the code. Suggest ways to implement idea. Then produce code that implements based those suggestions",
    "Error proof":"Check this code(context) for errors. Suggest ways to improve. Then produce code that addresses those suggestions",
    "Error correct":"I will give you a prompt with code(context), and the error produced by the code. Determine the issue. Suggest ways to fix it. Then produce code that addresses those suggestions",
    "Annotate":"I will give you code(context). You will annotate the code with comments explaining its function. You will also improve upon preexisting comments in the code. Respond only in code",
    "Explain":"I will give you a code, output of a code, or a piece of text. Explain in long detail what each part of the code, output or text means.",
    "Demo":" I will provide you one or multiple ideas or concepts, and the context will be either R or Python. Create a complete and executeable code that showcases that idea or concept. Respond only in code, and all descriptions of the code will be in the form of comments",
    "Mutate":"I will provide you code. Add a new cool function of your choosing to the code(dont add color). Respond in complete code and describe what new feature does in comments. The new code must be at least slightly different from the original",
    "Clean":"I will provide you code. Determine if it contains any unused, redundant, poorly implemented, or error prone code. If there is input and context, perform the inputted cleaning action requested on the code provided",
    "Help":"I will prove you code, and ask a troubleshooting question about it",
    "Criticize":"I will provide you code, with or without a description of its intended function. You  will assess the code's functions, criticize it, and suggest improvements",
    "Test": "Create unit tests for the given code to ensure it functions as expected and identify any potential bugs that need to be fixed",
    "Possibilities":"I will provide you code. You will determine its functions, and suggest possibilities for new functions and ideas for the code",
    "Integrate":"I will give you 2 codes. You will integrate the first code with the second code.",
    "Optimize": "Evaluate the given code for performance bottlenecks and suggest or implement optimizations to improve its efficiency",
    "Convert":"I will give you a script and the name of a coding language. You will take the script and convert it to the provided coding language",
    "Traceback":"I will provide you the output of a traceback() command. You will describe what issue is in the traceback output, and what the solution might be",
    "General":"I will give you code, and a general request or question"
}

data_systems = {
    "--":"--",
    "Data summary":"I will give you data(ie a table). You will summarize the data, report outliers, mean, noticeable trends, and steps to make data tidy and read for use with R",
    "Implications":"I will provide you with some statistical output/result. You will determine what the results imply about the data",
    "BLAST Title":"""
I will give you the title of a blast alignment(i.e. 'Alignment: gi|1950600667|dbj|AP018365.1| Streptomyces sp. SN-593'). You will interpret the meaning of the title like so:
"Alignment: gi|1747255792|gb|AC279172.1| Callithrix jacchus BAC CH259-208K15 from chromosome Y, complete sequence" refers to a blast alignment result involving the following information:

1. "gi|1747255792": The GenInfo Identifier (gi) number, which is a unique identifier for .....
2. "gb|AC279172.1": The GenBank (gb) Accession number, which consists of ....
3. "Callithrix jacchus": The scientific name of the .....
4. "BAC CH259-208K15": The identifier of a BAC (Bacterial Artificial Chromosome) clone....
5. "from chromosome Y": Indicates that .....
6. "complete sequence": Indicates that ....""",
    "General":"I will ask you a question about data"
}


def button_clicked(option, group):
    # Global variables for maintaining previous input, previous response, and direct input throughout the program
    global previous_input
    global previous_response
    global direct_input

    # Assigning the selected_system according to the specified group value
    if group == 1:
        selected_system = system = code_systems[option]
    if group == 2:
        selected_system = system = text_systems[option]
    if group == 3:
        selected_system = system = data_systems[option]
        print('ok')

    # Find the button_name by iterating over code_systems, text_systems, and data_systems dict
    button_name = None
    systems = [code_systems, text_systems, data_systems]
    for system in systems:
        for k, v in system.items():
            if v == selected_system:
                button_name = k

    # Print the selected button's name with a prefix "button-"
    print("button-" + button_name)

    # Maintain the value of direct_input
    direct_input = direct_input

    # Get the current date and time and format it
    current_date_time = datetime.datetime.now()
    formatted_date_time = current_date_time.strftime("%m/%d/%Y %H:%M:%S")

    # Print a separator and the formatted date-time
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Datetime:", formatted_date_time)

    # Prepare the prompt and the context based on input_entry and context_entry
    prompt = "Input: " + input_entry.get()
    context = "Context: " + context_entry.get()

    # Check for input_entry and context_entry values and update the prompt accordingly
    if context_entry.get() != None:
        if input_entry.get() != None:
            prompt = "Input: " + input_entry.get() + ". Context: " + context_entry.get()
        elif input_entry.get() == None:
            if loaded_prompt != None:
                prompt = "Input: " + loaded_prompt + ". Context: " + context_entry.get()
            else:
                prompt = context_entry.get()
    else:
        if loaded_prompt != None:
            prompt = "Input: " + loaded_prompt
        else:
            prompt = input_entry.get()

    # Append direct input to the prompt if it is not None
    if direct_input != None:
        prompt = prompt + direct_input

    # Run run_gpt function with the prepared prompt, selected_system, and button_name
    run_gpt(prompt, selected_system, button_name)

def save_to_history(formatted_date_time, prompt, response, system, button_name=None):
    with open("history.txt", "a", encoding="utf-8") as history_file:
        history_file.write("\n" + "===" * 30 + "\n")
        history_file.write("Datetime: " + formatted_date_time + "\n\n")
        if button_name is not None:
            history_file.write("System:\n" +button_name+ ": "+ system + "\n\n")
        else:        
            history_file.write("System:\n" + system + "\n\n")
        history_file.write("Prompt:\n" + prompt + "\n\n")
        history_file.write("Response:\n" + response)

def run_gpt(prompt, system, button_name):
    # Get the current date and time
    current_date_time = datetime.datetime.now()
    # Format the date and time as a string
    formatted_date_time = current_date_time.strftime("%m/%d/%Y %H:%M:%S")

    # Print the system message and user prompt
    print(system)
    print(prompt)
    # Print a separator line
    print("_______________________________________________________________________________________________________________________________________________________________")

    # Record the start time of GPT processing
    start_time = time.time()

    # Call the GPT model with the system and user messages
    completion = openai.ChatCompletion.create(
      model="gpt-4", #gpt-3.5-turbo
      messages=[{"role": "system", "content": system},
                {"role": "user", "content": prompt}]
    )
    
    # Retrieve the GPT response content
    response = completion["choices"][0].message["content"]

    # Calculate and print the elapsed time for GPT processing
    elapsed_time = time.time() - start_time
    print(response)
    print("================================================================================================================================================================")
    print("\n")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    # Clear the output_text widget and display the GPT response and elapsed time
    output_text.delete(1.0, tk.END)
    response_with_time = f"{response}\n\nElapsed time: {elapsed_time:.2f} seconds"
    output_text.insert(tk.END, response_with_time)

    # Save the interaction with the GPT response, elapsed time, and other details to the history file
    save_to_history(formatted_date_time, prompt, response_with_time, system, button_name)

def load_file_to_input():
    # Set loaded_prompt as a global variable so it can be used outside this function
    global loaded_prompt

    # Open a file dialog to allow the user to choose a file
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])

    # Check if a file has been selected
    if file_path:
        # Open the selected file and read its content
        with open(file_path, 'r', errors='replace') as file:
            content = file.read()
            # Print the content of the file
            print(content)
            # Clear the input_entry widget
            input_entry.delete(0, 'end')
            # Store the content of the file in loaded_prompt
            loaded_prompt = content
                
def clear_context_entry():
    # Clear the context_entry widget
    context_entry.delete(0, 'end')

def paste_into_entry(entry):
    # Paste the clipboard content into the given entry widget
    entry.insert(tk.END, root.clipboard_get())

def clear_input_entry():
    # Set direct_input as a global variable so it can be used outside this function
    global direct_input

    # Clear the input_entry widget
    input_entry.delete(0, 'end')
    # Reset direct_input
    direct_input = None
    
def copy_output_to_clipboard():
    # Clear the current content of the clipboard
    root.clipboard_clear()
    # Append the content of the output_text widget to the clipboard
    root.clipboard_append(output_text.get(1.0, tk.END))

def input_multiline() -> str:
    # Initialize a list to store the input lines
    lines = []
    # Prompt the user for input and inform them how to finish input
    print("Enter your text. Type '@END' to finish:")
    # Read lines of input until '@END' is received
    while True:
        line = input()
        if line.strip() == '@END':
            break
        lines.append(line)
    # Combine the lines into a single string with line breaks
    return '\n'.join(lines)

def get_direct_input():
    # Set direct_input as a global variable so it can be used outside this function
    global direct_input
    # Call the input_multiline function to receive a multiline string
    direct_input = input_multiline()
    # If direct_input has been received, print a success message
    if direct_input != None:
        print("Direct input saved!")



root = tk.Tk()

root.title("GPT Go")
# Setting up style
style = ttk.Style()
#style.theme_use('xpnative')
# Import the tcl file with the tk.call method

# Set the theme with the theme_use method
style.theme_use('clam')  
style.configure('.', font=('Helvetica', 10))
style.configure('TLabel', padx=10, pady=10)
style.configure('TButton', padx=5, pady=5, width=10)
style.map('TButton', background=[('pressed', '#E1E1E1'), ('active', '#E1E1E1')])

# Create a Tkinter GUI with multiple frames and dropdown menus

# Create a frame for the header section and place it in the root window
header_frame = tk.Frame(root)
header_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
header_frame.columnconfigure(0, weight=1)

# Create a frame for the input section and place it in the root window
input_frame = tk.Frame(root)
input_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
input_frame.columnconfigure(1, weight=1)

# Create a frame for the context section and place it in the root window
context_frame = tk.Frame(root)
context_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
context_frame.columnconfigure(1, weight=1)

# Create a frame for the output section and place it in the root window
output_frame = tk.Frame(root)
output_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
output_frame.columnconfigure(0, weight=1)

# Add a label to the header frame asking user to select a request type
header_label = ttk.Label(header_frame, text="Please select a request type:")
header_label.grid(row=0, column=0, sticky='w')

# Create StringVar objects to hold the selected options from dropdown menus
header_var1 = tk.StringVar(header_frame)
header_var1.set("Please select a code system")
header_var2 = tk.StringVar(header_frame)
header_var2.set("Please select a text system")
header_var3 = tk.StringVar(header_frame)
header_var3.set("Please select a data system")

# Add a label to the header frame for the 'Code Systems' dropdown menu
code_systems_label = ttk.Label(header_frame, text="Code Systems:")
code_systems_label.grid(row=1, column=0, sticky='e')

# Create the 'Code Systems' dropdown menu with options from 'code_systems' dictionary
header_option_menu1 = ttk.OptionMenu(header_frame, header_var1, *code_systems.keys(), command=lambda option: button_clicked(option, group=1))
header_option_menu1.grid(row=1, column=1, sticky='e', padx=5)

# Creating a tkinter ttk.Label for "Text Systems" in header_frame
text_systems_label = ttk.Label(header_frame, text="Text Systems:")
# Positioning the label using grid layout
text_systems_label.grid(row=1, column=2, sticky='e')

# Creating a tkinter ttk.OptionMenu for text_systems in header_frame
header_option_menu2 = ttk.OptionMenu(header_frame, header_var2, *text_systems.keys(), command=lambda option: button_clicked(option, group=2))
# Positioning the option menu using grid layout
header_option_menu2.grid(row=1, column=3, sticky='e', padx=5)

# Creating a tkinter ttk.Label for "Data Systems" in header_frame
data_systems_label = ttk.Label(header_frame, text="Data Systems:")
# Positioning the label using grid layout
data_systems_label.grid(row=1, column=4, sticky='e')

# Creating a tkinter ttk.OptionMenu for data_systems in header_frame
header_option_menu3 = ttk.OptionMenu(header_frame, header_var3, *data_systems.keys(), command=lambda option: button_clicked(option, group=3))
# Positioning the option menu using grid layout
header_option_menu3.grid(row=1, column=5, sticky='e', padx=5)

# Creating a tkinter ttk.Label for user input in input_frame
input_label = ttk.Label(input_frame, text="Please enter your input:")
# Positioning the label using grid layout
input_label.grid(row=0, column=0)

# Creating a tkinter ttk.Entry for user input in input_frame
input_entry = ttk.Entry(input_frame)
# Positioning the entry using grid layout
input_entry.grid(row=0, column=1, sticky='ew', padx=5)

# Creating a tkinter ttk.Button for pasting text into input_entry
input_paste_button = ttk.Button(input_frame, text="Paste", command=lambda: paste_into_entry(input_entry))
# Positioning the button using grid layout
input_paste_button.grid(row=0, column=2, padx=5)

# Creating a tkinter ttk.Button for loading a file to input_entry
load_file_button = ttk.Button(input_frame, text="Load File", command=load_file_to_input)
# Positioning the button using grid layout
load_file_button.grid(row=0, column=3, padx=5)

# Creating a tkinter ttk.Button for clearing the input_entry
clear_context_button = ttk.Button(input_frame, text="Clear Input", command=clear_input_entry)
# Positioning the button using grid layout
clear_context_button.grid(row=0, column=4, padx=5)

# Creating a tkinter ttk.Label for context input in context_frame
context_label = ttk.Label(context_frame, text="Please enter your contextual(ie code, text) info:")
# Positioning the label using grid layout
context_label.grid(row=0, column=0)

# Creating a tkinter ttk.Entry for context input in context_frame
context_entry = ttk.Entry(context_frame)
# Positioning the entry using grid layout
context_entry.grid(row=0, column=1, sticky='ew', padx=5)

# Creating a tkinter ttk.Button for pasting text into context_entry
context_paste_button = ttk.Button(context_frame, text="Paste", command=lambda: paste_into_entry(context_entry))
# Positioning the button using grid layout
context_paste_button.grid(row=0, column=2, padx=5)

# Creating a tkinter ttk.Button for clearing the context_entry
clear_context_button = ttk.Button(context_frame, text="Clear Context", command=clear_context_entry)
# Positioning the button using grid layout
clear_context_button.grid(row=0, column=3, padx=5)

# Creating a tkinter ttk.Button for direct input
direct_input_button = ttk.Button(input_frame, text="Direct Input", command=get_direct_input)
# Positioning the button using grid layout
direct_input_button.grid(row=0, column=5, padx=5)

# Creating a tkinter ttk.Label for displaying output in output_frame
output_label = ttk.Label(output_frame, text="Output will be displayed here.", wraplength=500)
# Positioning the label using grid layout
output_label.grid(row=0, column=0, sticky='w')

# Creating a tkinter ttk.Scrollbar for output_frame
output_scrollbar = ttk.Scrollbar(output_frame)
# Positioning the scrollbar using grid layout
output_scrollbar.grid(row=1, column=1, sticky="ns")

# Creating a tkinter Text widget for displaying output
output_text = tk.Text(output_frame, wrap="word", yscrollcommand=output_scrollbar.set, bg="#F7F7F7", font=('Helvetica', 12))
# Positioning the output text using grid layout
output_text.grid(row=1, column=0, sticky="nsew")
# Configuring the output scrollbar to work with output_text
output_scrollbar.config(command=output_text.yview)

# Creating a tkinter ttk.Button for copying output to clipboard
copy_output_button = ttk.Button(output_frame, text="Copy Output", command=copy_output_to_clipboard)
# Positioning the button using grid layout
copy_output_button.grid(row=2, column=0, pady=5)

# Configuring output_frame row to resize with window
output_frame.rowconfigure(1, weight=1)

# Main event loop to run the tkinter application
root.mainloop()
