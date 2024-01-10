import tkinter as tk
import openai
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

def chat_with_model(prompt):
    # Set up OpenAI API credentials
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

def cutlength(text):
    if( len(text) > 1500):
        text = text[:1500]
    return text

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

class TextDataset(Dataset):
    def __init__(self, real_texts, fake_texts):
        self.real_texts = real_texts
        self.fake_texts = fake_texts

    def __len__(self):
        return len(self.real_texts) + len(self.fake_texts)

    def __getitem__(self, index):
        if index < len(self.real_texts):
            answer = self.real_texts[index]
            label = 0
        else:
            answer = self.fake_texts[index - len(self.real_texts)]
            label = 1
        return answer, label

def change(x):
    y = []
    for z in x:
        if(z == 'Human'):
            y.append(0)
        else:
            y.append(1)
            
    return y

def loader(batch_size=32, prompt = 'p1', train_name = 'World'):
    if( type(prompt) == int):
        real_data, fake_data = Corpus_prompt( prompt=prompt, train_name = train_name)
    else:
        real_data, fake_data = Corpus_all( train_name = train_name)
    real_train = real_data[0:len(real_data) - 400]
    real_valid = real_data[len(real_data) - 400:len(real_data) - 250]
    real_test = real_data[len(real_data) - 250:]

    fake_train = fake_data[0:len(fake_data) - 400]
    fake_valid = fake_data[len(fake_data) - 400:len(fake_data) - 250]
    fake_test = fake_data[len(fake_data) - 250:]

    weight = torch.cat(
        [len(fake_train) / len(real_train) * torch.ones(len(real_train)), torch.ones(len(fake_train))])
    train_sampler = WeightedRandomSampler(weight, len(real_train) + len(fake_train), replacement=False)

    train_dataset = TextDataset(real_train, fake_train)
    train_loader = DataLoader(train_dataset, batch_size, sampler=train_sampler, num_workers=0)

    test_dataset = TextDataset(real_test, fake_test)
    test_loader = DataLoader(test_dataset, 1, shuffle=True, num_workers=0)

    valid_dataset = TextDataset(real_valid, fake_valid)
    valid_loader = DataLoader(valid_dataset, 1, shuffle=True, num_workers=0)

    return train_loader, valid_loader, test_loader

def Corpus_prompt(prompt, train_name):

    fake_data = []
    real_data = []

    jsonl_data = load_jsonl('/Users/amyliu/Documents/hshsp/project/week5/detect/all_datav3.jsonl')
    # jsonl_data = load_jsonl('/mnt/home/wangw116/amy/week5/detect/all_datav3.jsonl')

    for message in jsonl_data:
        if( message['name'] == train_name):
            if( message['label'] == 0):
                real_data.append(message['text'])
            elif( message['prompt'] == prompt):
                fake_data.append(message['text'])

    fake_data = cut_data(fake_data)
    real_data = cut_data(real_data)
        
    return real_data, fake_data

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def process_spaces(story):
    try:
        story = story[0].upper() + story[1:]
    except:
        print(story)
    story =  story.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').replace(
        '<br />', '').strip()
    story = remove_last_sentence(story)
    return story

def strip_newlines(text):
    return ' '.join(text.split())

def cut_data(data):
    data = [x.strip() for x in data]
    data = [strip_newlines(x) for x in data]
    data = [process_spaces(x) for x in data]
    long_data= []
    for i in range(len(data)):
        x = data[i]
        if( len(x.split()) > 32 ):
            long_data.append(x)
    random.shuffle(long_data)
    return long_data

def remove_last_sentence(paragraph):
    # Split the paragraph into sentences
    sentences = paragraph.split('. ')
    # Check if the last sentence ends with '.', '?', or '!'
    last_sentence = sentences[-1]
    if last_sentence.endswith(('.', '?', '!')):
        return paragraph  # Return the original paragraph if last sentence is not ended by '.', '?', or '!'
    else:
        if len(sentences) > 1:
            sentences.pop()
        # Join the remaining sentences
        modified_paragraph = '. '.join(sentences) +'.'
        return modified_paragraph



num = 10
# per prompt

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
####################################################################IMDB

for i in range(3):
    _,_,test_loader = loader(prompt= i+1, train_name='IMDb')
    for i in range(num):
        text, label = next(iter(test_loader))
        answer = text[0]
        all_labels.append(label.item())
        q = {
            "text": "Review: " + cutlength(answer),
            "options": ["Human", "ChatGPT"],
            "variables": []}
        questions.append(q)


####################################################################
####################################################################
####################################################################QA

for i in range(3):
    _,_,test_loader = loader(prompt= i+1, train_name='Amazon')
    for i in range(num):
        text, label = next(iter(test_loader))
        answer = text[0]
        all_labels.append(label.item())
        q = {
            'text': "Review: " + cutlength(answer),
            "options": ["Human", "ChatGPT"],
            "variables": []}
        questions.append(q)

####################################################################
####################################################################
####################################################################QA

for i in range(3):
    _,_,test_loader = loader(prompt= i+1, train_name='Eli5')
    for i in range(num):
        text, label = next(iter(test_loader))
        answer = text[0]
        all_labels.append(label.item())
        q = {
            'text': "Answer: " + cutlength(answer),
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
all_results = change(all_results)
print(all_results)
print(all_labels)

order = ["imdb", "amazon", 'eli5']

for i in range(3):
    for j in range(3):
        curr_results = all_results[i*num:i*num + num]
        curr_labels = all_labels[i*num:i*num + num]

        f1 = f1_score(curr_labels, curr_results)

        print('.....................' + order[i] + "_" + str(j+1) + '.....................')
        print("f1: " + str(f1))

print("DONE !!!!!!!")