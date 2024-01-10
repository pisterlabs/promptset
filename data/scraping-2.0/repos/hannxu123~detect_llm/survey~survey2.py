import tkinter as tk
import openai
from dataset2 import loader

def chat_with_model(prompt):
    # Set up OpenAI API credentials
    openai.api_key = "sk-W0kcNhgjghjARAz2GWI8sdfshfjjhlbkFJ8jgO935Imc3nqer4Y"

    # Define the ChatGPT parameters
    model = "gpt-3.5-turbo"
    max_tokens = 300

    # Generate a response from the ChatGPT model
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )

    # Extract and return the reply from the model's response
    reply = response.choices[0].message.content.strip()
    return reply


def send_message():
    user_input = entry.get()
    entry.delete(0, tk.END)

    chat_history_text = chat_history.get("1.0", tk.END)
    chat_history_text += "You: " + user_input + "\n"
    chat_history.delete("1.0", tk.END)
    chat_history.insert(tk.END, chat_history_text)

    # Add the user input to the prompt
    prompt = chat_history_text + "User: " + user_input

    # Get the model's reply
    reply = chat_with_model(prompt)
    chat_history_text += "ChatGPT: " + reply + "\n"
    chat_history.delete("1.0", tk.END)
    chat_history.insert(tk.END, chat_history_text)
    chat_history.see(tk.END)


def start_survey():
    welcome_frame.pack_forget()
    survey_frame.pack()


def submit():
    selected_options = []
    for var, option in zip(current_question["variables"], current_question["options"]):
        if var.get() == 1:
            selected_options.append(option)
    responses.append({"question": current_question["text"], "selected_options": selected_options})

    if current_question_index < len(questions) - 1:
        next_question()
    else:
        print_responses()
        root.quit()  # Close the window after all questions are answered


def next_question():
    global current_question_index, current_question
    current_question_index += 1
    current_question = questions[current_question_index]

    question_label.config(text=current_question["text"])

    for var in current_question["variables"]:
        var.set(0)

    for checkbox in checkboxes:
        checkbox.pack_forget()

    checkboxes.clear()

    for option in current_question["options"]:
        var = tk.IntVar()
        current_question["variables"].append(var)
        checkbox = tk.Checkbutton(survey_frame, text=option, variable=var, anchor = 'w', justify=tk.LEFT)
        checkbox.pack()
        checkboxes.append(checkbox)

    submit_button.pack_forget()
    submit_button.pack(side='bottom', pady=10)


def print_responses():
    print("Selected options:")
    for response in responses:
        print(response["selected_options"])
        all_results.append(response["selected_options"][0])

######################### main loop
root = tk.Tk()
root.title("Survey")
root.geometry("1200x800")  # Set the width and height as desired

# Create welcome page
welcome_frame = tk.Frame(root)
welcome_label = tk.Label(welcome_frame, text="Welcome! ChatGPT is a powerful tool for various language tasks. "
                                    "\n The following sentences can be written by ChatGPT! \n Use your "
                                  "best knowledge to distinguish. \n You can also use the tool below to request a ChatGPT. Please click 'Start Survey'")
welcome_label.pack(padx=20, pady=20)
start_button = tk.Button(welcome_frame, text="Start Survey", command=start_survey)
start_button.pack(padx=20, pady=10)
welcome_frame.pack()

# Create survey page
survey_frame = tk.Frame(root)

# Create a text widget to display the chat history
chat_history = tk.Text(root, height=9, width=80)
chat_history.pack()

# Create an entry widget for user input
label = tk.Label(text="Your question goes here:")
label.pack()
entry = tk.Entry(root, width=80)
entry.pack()

# Create a button to send the user's message
send_button = tk.Button(root, text="Ask ChatGPT", command=send_message)
send_button.pack()



questions = []
all_labels  = []
all_results = []

####################################################################
####################################################################
####################################################################News
test_loader = loader('ivy')
for i in range(5):
    _, text, label = next(iter(test_loader))
    answer = text[0]
    # question = question[0]
    all_labels.append(label.item())
    q = {
        "text": " The article: \" " + answer + "\"",
        "options": ["Human", "ChatGPT"],
        "variables": []}
    questions.append(q)


####################################################################
####################################################################
####################################################################News
test_loader = loader('world')
for i in range(5):
    _, text, label = next(iter(test_loader))
    answer = text[0]
    # question = question[0]
    all_labels.append(label.item())
    q = {
        "text": " The news: \" " + answer + "\"",
        "options": ["Human", "ChatGPT"],
        "variables": []}
    questions.append(q)

####################################################################
####################################################################
####################################################################QA
test_loader = loader('eli5')
for i in range(5):
    question, text, label = next(iter(test_loader))
    answer = text[0]
    question = question[0]
    all_labels.append(label.item())
    q = {
        "text": "Answer the question: \"" + str(question) + "\". Answer: \"" + answer + "\"",
        "options": ["Human", "ChatGPT"],
        "variables": []}
    questions.append(q)

####################################################################
####################################################################
####################################################################IMDB

test_loader = loader('imdb')
for i in range(5):
    question, text, label = next(iter(test_loader))
    answer = text[0]
    question = question[0]
    all_labels.append(label.item())
    q = {
        "text": "Movie name: " + str(question) + ". Review: \"" + answer + "\"",
        "options": ["Human", "ChatGPT"],
        "variables": []}
    questions.append(q)

####################################################################
####################################################################
####################################################################Amazon

test_loader = loader('amazon')
for i in range(5):
    question, text, label = next(iter(test_loader))
    answer = text[0]
    question = question[0]
    all_labels.append(label.item())
    q = {
        "text": "Amazon product: " + question + ". User review: \"" + answer + "\"",
        "options": ["Human", "ChatGPT"],
        "variables": []}
    questions.append(q)


current_question_index = 0
current_question = questions[current_question_index]

question_label = tk.Label(survey_frame, text=current_question["text"], wraplength=600)
question_label.pack(anchor="w")

checkboxes = []
for option in current_question["options"]:
    var = tk.IntVar()
    current_question["variables"].append(var)
    checkbox = tk.Checkbutton(survey_frame, text=option, variable=var)
    checkbox.pack(anchor='w')
    checkboxes.append(checkbox)


responses = []  # Initialize the responses list

submit_button = tk.Button(survey_frame, text="Submit", command=submit)
submit_button.pack(side='bottom', pady=10)

root.mainloop()

###############################
print('################################')
print(all_results)
print(all_labels)
