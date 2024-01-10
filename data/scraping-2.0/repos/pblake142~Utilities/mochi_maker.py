# A barebones utility that can generate a mochi-ready EDN file using a TKInter UI

import tkinter as tk
import json
import re
import time
import tkinter.messagebox
from openai import OpenAI
from tkinter import *

MOCHI_QUESTIONS_GENERATOR_SYSTEM_PROMPT = """
You are a question generation program. When you are provided with a user message, write a list of all questions about the information conatined in the user message. No two questions should ask about the same information from the user message. Print the questions in a numbered format.
"""
MOCHI_ANSWER_GENERATOR_SYSTEM_PROMPT = "You are an answer generation program. When you are provided with a user message, it will contain text before a list of numbered questions. The numbered questions will begin after the '-----' string. Write answers for each of the numbered questions using the text preceding it."

client = OpenAI()

# Core function to interact with OpenAI API
def openAIGPTCall(system_prompt,user_prompt, model="gpt-4"):
  print("Calling OpenAI API.")  
  
  message_call = {
     "messages": [{
        "role": "system",
        "content": system_prompt
     },
     {
        "role": "user",
        "content": user_prompt
     }
     ],
     "model": model,
  }

  start_time = time.time()
  response = client.chat.completions.create(**message_call)
  elapsed_time = (time.time() - start_time) * 1000
  cost_factor = 0.06 if model == "gpt-4-0613" else 0.002 # Update with pricing
  cost = cost_factor * (response.usage.total_tokens / 1000)
  message = response.choices[0].message.content.strip()
  return message, cost, elapsed_time

# Function to generate questions
def question_generator(text):
  system_prompt = MOCHI_QUESTIONS_GENERATOR_SYSTEM_PROMPT
  user_prompt = text

  print("Passing text to generate questions.")
  questions,question_cost,question_time = openAIGPTCall(system_prompt,user_prompt, model="gpt-4-0613")
  print("Questions generated.")
  return questions, question_cost, question_time

# Function to generate answers
def answer_generator(text, questions):
  system_prompt = MOCHI_ANSWER_GENERATOR_SYSTEM_PROMPT
  user_prompt = text+'-----'+questions

  print(f"user_prompt: {user_prompt}")

  print("Passing text and questions to generate answers.")
  answers, answer_cost, answer_time = openAIGPTCall(system_prompt,user_prompt, model="gpt-4-0613")
  print("Answers generated.")
  return answers, answer_cost, answer_time

# Function that pairs the generated questions and answers
def content_generator(questions,answers):
  
  question_list = re.findall(r'(\d{1,2})\. (.*?)(?=\n\d{1,2}\.|$)', questions, re.DOTALL)
  answer_list = re.findall(r'(\d{1,2})\. (.*?)(?=\n\d{1,2}\.|$)', answers, re.DOTALL)

  # Creating a txt file that I can use to debug
  with open('questions.txt','w', encoding='utf-8') as f:
    f.write(questions)

  qa_list = [{'content': f"{q[1]}\n---\n{a[1]}"} for q, a in zip(question_list, answer_list) if q[0] == a[0]]

  # Creating a txt file that I can use to debug
  with open('qa_list.txt','w', encoding='utf-8') as f:
    f.write(str(qa_list))

  return qa_list

# Function that adds the question/answer pairs to the EDN file
def add_content(text):
  global edn, total_cost, total_time
  
  questions, question_cost, question_time = question_generator(text)
  answers, answer_cost, answer_time = answer_generator(text,questions)
  qa_list = content_generator(questions,answers)

  for pair in qa_list:
    edn += "{:content \"" + pair['content'] + "\"}\n"

  total_cost += question_cost+answer_cost
  total_time += question_time+answer_time

# Initialize the variable for storing the information
qa_pairs = []
edn = ""
total_cost = 0
total_time = 0

# Function used to generate the Q/A pairs and display in GUI
def generate():
    global edn, total_cost, total_time

    text = input_box.get('1.0', tk.END).strip()
    if text:
        questions, question_cost, question_time = question_generator(text)
        answers, answer_cost, answer_time = answer_generator(text, questions)
        qa_list = content_generator(questions, answers)

        # Add the content to the display box and to the JSON object
        for pair in qa_list:
            display_box.insert(tk.END, pair['content'] + '\n')
            qa_pairs.append(pair['content'])

            # Update the global variables
            edn += "{:content \"" + pair['content'].replace("\n", "\\n").replace("\"", "\\\"") + "\"}\n"

        total_cost += question_cost + answer_cost
        total_time += question_time + answer_time

        # Clear the input box
        input_box.delete('1.0', tk.END)

# Function that saves any edits made to the display box
def save_edits():
    # Get the current text in the display box
    text = display_box.get('1.0', tk.END).strip()

    # Split the text into individual question and answer pairs
    pairs = text.split('\n---\n')

    # Clear the existing question and answer pairs in the JSON object
    qa_pairs.clear()
    
    # Clear the 'edn' variable
    global edn
    edn = ""
    
    # Add the edited pairs to the JSON object and update the 'edn' variable
    for pair in pairs:
        qa_pairs.append(pair)
        edn += "{:content \"" + pair.replace("\n", "\\n").replace("\"", "\\\"") + "\"}\n"
    
    # Let the user know the changes have been saved
    tkinter.messagebox.showinfo("Edits Saved", "The changes have been successfully saved.")

def generate_edn():
    global edn

    # Get the name of the deck
    deck_name = deck_name_entry.get().strip()
    if not deck_name:
        tkinter.messagebox.showwarning("No Deck Name", "Please enter a name for the deck before generating the EDN file.")
        return

    # Format the EDN string with the deck name and question-answer pairs
    edn_file_content = f'{{:version 2\n :decks [{{:name "{deck_name}" \n :cards [{edn}]}}]}}'

    with open('qa_pairs.json', 'w') as f:
        json.dump(qa_pairs, f, indent=4)

    # Write the EDN string to a file
    with open('data.edn', 'w', encoding='utf-8') as f:
        f.write(edn_file_content)

    # Clear the global EDN string and the display box
    edn = ""
    display_box.delete('1.0', tk.END)

    tkinter.messagebox.showinfo("EDN Generated", "The EDN file has been successfully generated.")

from tkinter import filedialog

def generate_edn_from_json():
    # Open a file dialog and get the chosen file's path
    filepath = filedialog.askopenfilename(filetypes=[('JSON files', '*.json')])

    if filepath:
      # Read the JSON file and convert the data to the 'edn' string format
      with open(filepath, 'r') as f:
          data = json.load(f)
          edn = ""
          for pair in data:
              edn += "{:content \"" + pair.replace("\n", "\\n").replace("\"", "\\\"") + "\"}\n"

      # Write the 'edn' string to an EDN file
      with open('data.edn', 'w', encoding='utf-8') as f:
          f.write(edn)

    else:
      tkinter.messagebox.showwarning("No File Selected", "Please select a JSON file before generating the EDN file.")


# Initialize GUI
root = tk.Tk()
root.title("Mochi Maker")

# Add components to GUI
header = tk.Label(root, text="Mochi Maker", font=("Arial", 24))
header.pack()

deck_name_label = tk.Label(root, text="Deck Name", font=("Arial", 16))
deck_name_label.pack()
deck_name_entry = tk.Entry(root, font=("Arial", 16))
deck_name_entry.pack()

input_label = tk.Label(root, text="Input Text", font=("Arial", 16))
input_label.pack()
input_box = Text(root, width=40, height=10)
input_box.pack()

generate_button = tk.Button(root, text="Generate", command=generate)
generate_button.pack()

save_button = tk.Button(root, text="Save Edits", command=save_edits)
save_button.pack()

display_label = tk.Label(root, text="Question Display", font=("Arial", 16))
display_label.pack()
display_box = Text(root, width=40, height=10)
display_box.pack()

generate_edn_button = tk.Button(root, text="Generate EDN", command=generate_edn)
generate_edn_button.pack()

from_json_button = tk.Button(root, text="Generate EDN from JSON", command=generate_edn_from_json)
from_json_button.pack()


root.mainloop()
