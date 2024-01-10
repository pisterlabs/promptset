import os
import json
import yaml
from time import time
import openai
import inquirer  

def generate_summary(formatted_consensus_messages):
    summary = []
    for message in formatted_consensus_messages:
        summary.append(message['content'])
    return ' '.join(summary)



def save_yaml(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as file:
        yaml.dump(data, file, allow_unicode=True)

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
        return infile.read()

def get_persona_traits(filepath='persona_traits.json'):
    with open(filepath, 'r') as f:
        return json.load(f)

def chatbot(messages, model, temperature=0.9):
    openai.api_key = open_file('key_openai.txt')
    response = openai.ChatCompletion.create(model=model, messages=messages, temperature=temperature)
    text = response['choices'][0]['message']['content']
    return text

def select_personalities():
    questions = [
        inquirer.Checkbox('selected_personalities',
                          message="Which personalities do you want to include?",
                          choices=available_personalities,
                          ),
    ]
    answers = inquirer.prompt(questions)
    return answers['selected_personalities']


persona_traits_data = get_persona_traits()['MBTI Personality Types']
available_personalities = list(persona_traits_data.keys())
selected_personalities = select_personalities()  # New line to select personalities
issue = input("Describe an issue: ")

print("Creating personas...")
generated_files = []
time_str = str(time())
folder_path = f"personas/{time_str}"

# Create the folder if it doesn't exist
os.makedirs(folder_path, exist_ok=True)
for personality_type in selected_personalities:
    traits = persona_traits_data[personality_type]
    system_with_traits = open_file('system_persona_role.txt').replace('<<PERSONA>>', traits)
    messages = [
        {'role': 'system', 'content': system_with_traits},
        {'role': 'user', 'content': f"What do you think this persona will think about {issue}?"}
    ]
    response = chatbot(messages, model="gpt-3.5-turbo")
    messages.append({'role': 'assistant', 'content': response})
    
    filepath = f"{folder_path}/{personality_type}_{time()}.yaml"  # Use the same folder_path
    save_yaml(filepath, {'personality': personality_type, 'issue': issue, 'messages': messages})
    generated_files.append(filepath)

consensus_messages = []
formatted_consensus_messages = []  # Initialize the variable here

print("Done. Press Enter to start the debate or enter a number between 1-10 to auto-accept.")
auto_accept_input = input()

synthesized_policy = open_file('system_synthesize_policy.txt')

try:
    for rounds in range(10):
        for idx, filepath in enumerate(generated_files):
            persona_data = yaml.safe_load(open_file(filepath))
            personality_acronym = filepath.split('/')[-1][:4]
            last_two_messages = consensus_messages[-2:] if len(consensus_messages) >= 2 else []

            # Load the synthesized policy from the file
            synthesized_policy = open_file('system_synthesize_policy.txt')
            
            if idx == 0 and rounds == 0:
                user_content = f"Please provide a brief and concise opening argument on the issue: {issue}"
            else:
                user_content = f"What's your take on {issue}?"
            
            round_messages = [
                {'role': 'system', 'content': open_file('system_persona_role.txt').replace('<<PERSONA>>', persona_data['personality'])},
                {'role': 'user', 'content': f"What are your thoughts on this proposed policy: {synthesized_policy}?"}
            ] + last_two_messages


            new_response = chatbot(round_messages, model="gpt-3.5-turbo" if rounds > 0 else "gpt-3.5-turbo")
            
            final_answer_content = ""
            
            if new_response:
                try:
                    response_json = json.loads(new_response)
                    final_answer_content = response_json.get('final_answer', '')
                except json.JSONDecodeError:
                    print(f"Unable to parse response as JSON: {new_response}")
            
            print(f"{personality_acronym}: {final_answer_content}\n")

            if not auto_accept_input.isnumeric():
                user_input = input("Press Enter for the next reply, 'x' to cancel, or a number to auto-accept for that many rounds: \n")
                
                if user_input.lower() == 'x':
                    raise KeyboardInterrupt
                elif user_input.isnumeric():
                    auto_accept_input = user_input
            else:
                if rounds >= int(auto_accept_input):
                    user_input = input("Press Enter for the next reply, 'x' to cancel, or a number to auto-accept for that many rounds: \n")
                    if user_input.lower() == 'x':
                        raise KeyboardInterrupt
                    elif user_input.isnumeric():
                        auto_accept_input = user_input

            consensus_messages.append({'role': 'assistant', 'content': final_answer_content})
            formatted_consensus_messages.append({'role': personality_acronym, 'content': final_answer_content})



except KeyboardInterrupt:
    print("Debate cancelled. Generating summary of the conversation...")

    # Generate a summary (Here, I'm assuming you have a function called `generate_summary` to do this)
    summary = generate_summary(formatted_consensus_messages)

    issue_filename = issue.replace(" ", "_")
    report_path = f"reports/{issue_filename}.yaml"

    # Load existing report if it exists
    existing_report = {}
    if os.path.exists(report_path):
        with open(report_path, 'r') as file:
            existing_report = yaml.safe_load(file)

    # Add the summary at the top of the report
    existing_report['summary'] = summary

    # Save the report back
    with open(report_path, 'w') as file:
        yaml.dump(existing_report, file, default_flow_style=False)

    print("Summary has been generated and added to the report.")

issue_filename = issue.replace(" ", "_")
save_yaml(f"reports/{issue_filename}.yaml", {'issue': issue, 'messages': formatted_consensus_messages})
