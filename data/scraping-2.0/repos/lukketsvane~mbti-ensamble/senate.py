import os
import json
import yaml
from time import time
import openai
import inquirer
import random

def generate_summary(messages):
    return ' '.join(message['content'] for message in messages)

def generate_consensus_summary(messages):
    synthesized_policy = open_file('summary.txt')
    messages = [{'role': 'system', 'content': synthesized_policy}, {'role': 'user', 'content': 'What is the consensus based on the personalities involved?'}] + messages
    return chatbot(messages)

def save_yaml(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as file:
        yaml.dump(data, file, allow_unicode=True)

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
        return infile.read()

def get_persona_traits(filepath='persona_traits.json'):
    with open(filepath, 'r') as f:
        return json.load(f)['MBTI Personality Types']

def chatbot(messages, model="gpt-4", temperature=0.8):
    openai.api_key = open_file('key_openai.txt')
    response = openai.ChatCompletion.create(model=model, messages=messages, temperature=temperature)
    return response['choices'][0]['message']['content']

def select_personalities(available_personalities):
    questions = [inquirer.Checkbox('selected_personalities', message="Which personalities do you want to include?", choices=available_personalities)]
    answers = inquirer.prompt(questions)
    return answers['selected_personalities']

def main():
    persona_traits_data = get_persona_traits()
    selected_personalities = select_personalities(list(persona_traits_data.keys()))
    issue = input("Describe an issue: ")
    issue_filename = issue.replace(" ", "_")
    generated_files, consensus_messages, formatted_consensus_messages = [], [], []

    folder_path = f"personas/{str(time())}"
    os.makedirs(folder_path, exist_ok=True)

    for personality_type in selected_personalities:
        traits = persona_traits_data[personality_type]
        system_with_traits = open_file('system_persona_role.txt').replace('<<PERSONA>>', traits)
        messages = [{'role': 'system', 'content': system_with_traits}, {'role': 'user', 'content': f"What do you think this persona will think about {issue}?"}]
        response = chatbot(messages)
        messages.append({'role': 'assistant', 'content': response})
        filepath = f"{folder_path}/{personality_type}_{time()}.yaml"
        save_yaml(filepath, {'personality': personality_type, 'issue': issue, 'messages': messages})
        generated_files.append(filepath)

    auto_accept_input = input("Done. Press Enter to start the debate or enter a number between 1-10 to auto-accept.")
    num_rounds = int(auto_accept_input) if auto_accept_input.isnumeric() else 10
    auto_accept_count = num_rounds if auto_accept_input.isnumeric() else 0

    try:
        for _ in range(num_rounds):
            for idx, filepath in enumerate(generated_files):
                persona_data = yaml.safe_load(open_file(filepath))
                personality_acronym = filepath.split('/')[-1][:4]
                last_four_messages = consensus_messages[-4:] if len(consensus_messages) >= 4 else consensus_messages
                round_messages = [{'role': 'system', 'content': open_file('system_persona_role.txt').replace('<<PERSONA>>', persona_data['personality'])}, {'role': 'user', 'content': f"What are your thoughts on this proposed policy: {issue}?"}] + last_four_messages
                new_response = chatbot(round_messages)
                print(f"\n{personality_acronym}: {new_response}")

                if auto_accept_count > 0:
                    auto_accept_count -= 1
                else:
                    user_input = input("\nPress Enter to approve and continue... or enter 'x' to save, summarize, and close: or enter 'c' to clarify: ")

                    if user_input.lower() == 'c':
                        clarification = input("USER: ")
                        round_messages.append({'role': 'user', 'content': clarification})
                        new_response = chatbot(round_messages)
                        print(f"\n{personality_acronym}: {new_response}")

                    elif user_input.lower() == 'x':
                        summary = generate_summary(formatted_consensus_messages)
                        consensus_summary = generate_consensus_summary(formatted_consensus_messages)
                        report_path = f"reports/{issue_filename}.yaml"
                        report_data = {'issue': issue, 'summary': summary, 'consensus_summary': consensus_summary, 'messages': formatted_consensus_messages}
                        save_yaml(report_path, report_data)
                        print("Summary and consensus summary have been generated and added to the report.")
                        return

                consensus_messages.append({'role': 'assistant', 'content': new_response})
                formatted_consensus_messages.append({'role': 'assistant', 'content': new_response})

        summary = generate_summary(formatted_consensus_messages)
        consensus_summary = generate_consensus_summary(formatted_consensus_messages)
        report_path = f"reports/{issue_filename}.yaml"
        report_data = {'issue': issue, 'summary': summary, 'consensus_summary': consensus_summary, 'messages': formatted_consensus_messages}
        save_yaml(report_path, report_data)
        print("Debate finished. Summary and consensus summary have been generated and added to the report.")
    except KeyboardInterrupt:
        print("Debate cancelled. Generating summary of the conversation...")
        summary = generate_summary(formatted_consensus_messages)
        consensus_summary = generate_consensus_summary(formatted_consensus_messages)
        report_path = f"reports/{issue_filename}.yaml"
        existing_report = yaml.safe_load(open(report_path, 'r')) if os.path.exists(report_path) else {}
        existing_report['summary'] = summary
        existing_report['consensus_summary'] = consensus_summary
        save_yaml(report_path, existing_report)
        print("Summary and consensus summary have been generated and added to the report.")

if __name__ == "__main__":
    main()
