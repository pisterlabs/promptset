import os
import openai
from dotenv import load_dotenv
from diagnostic import calculate_loss 
import json

# import data
with open('./chat/persona_template.json', 'r') as f:
    template_data = json.load(f)
template = template_data['template']
template_two = template_data['template_two']

with open('./chat/questions.json', 'r') as f:
    qa_data = json.load(f)
predefined_questions = qa_data['predefined_questions']
true_answers = qa_data['true_answers']
attack_questions = qa_data['attack_questions']
true_attack_answers = qa_data['true_attack_answers']

# Define Jack's initial persona and conversation history
jack_persona = "My name is Jack. I like to party. My major is business. I am in college. I like pizza. I am 22 years old."
susan_persona = " My name is Susan i like to knit.  and I teach architecture. i am a professor. I like fruits. I am 42 years old. Talk like a middle-aged woman."
conversation_history = []

# Number of conversation rounds
num_rounds = 5  # You can change this to the desired number of rounds

def openai_chat(messages):
    completions = openai.ChatCompletion.create(
        model="text-davinci-003",
        messages=messages,
        max_tokens=50,
    )

    message = completions.choices[0].text
    return message.strip()

# Start the conversation with Jack's persona description
conversation_history.append(("Jack Persona:", jack_persona))
conversation_history.append(("Susan:", "Hi Jack, how is your day going?"))

diagnostics_log = []

for round in range(num_rounds):
    # jack responds
    jack_response = openai_chat(conversation_history)
    conversation_history.append({"role": "Jack", "content": jack_response})

    for question, true_answer in zip(predefined_questions, true_answers):
        jack_diag_response = openai_chat(conversation_history + [{"role": "User", "content": question}])
        # calc loss 
        loss = calculate_loss(None, tokenizer, conversation_history, jack_diag_response, true_answer)

        # Add diagnostic data to log and CSV
        diagnostics_log.append({
            'question': question,
            'response': jack_diag_response,
            'loss': loss
        })
        print(diagnostics_log)

    # susan responds
    susan_response = openai_chat(conversation_history)
    conversation_history.append({"role": "Susan", "content": susan_response})
    


# Print the entire conversation history
print("\nFull Conversation History:")
for exchange in conversation_history:
    print(exchange[0], exchange[1])