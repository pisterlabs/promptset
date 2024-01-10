import logging
from flask import Flask, request, jsonify
import os
import openai
import csv

app = Flask(__name__)

# Configuring logging
logging.basicConfig(level=logging.ERROR)

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route('/generate_structure', methods=['POST'])
def generate_structure():
    data = request.json

    csv_file = data.get('csv_file')
    if not csv_file:
        return jsonify({"error": "CSV file not provided"}), 400

    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        csv_data = list(reader)

    # Define the basic Ansible role structure
    role_structure = [
        "tasks/main.yml",
        "handlers/main.yml",
        "vars/main.yml",
        "defaults/main.yml",
        "meta/main.yml",
        "templates/",
    ]

    # Define the prompts for each file
    prompts = {
        "tasks/main.yml": "Generate the main tasks YAML code for the Ansible role for {tool} from the repository {repo}. Start with the first task and end with the last one, without any explanation or markdown.",
        "handlers/main.yml": "Generate the handlers YAML code for the Ansible role for {tool} from the repository {repo}. Start with the first handler and end with the last one, without any explanation or markdown.",
        "vars/main.yml": "Generate the variables YAML code for the Ansible role for {tool} from the repository {repo}. Start with the first variable and end with the last one, without any explanation or markdown.",
        "defaults/main.yml": "Generate the default variables YAML code for the Ansible role for {tool} from the repository {repo}. Start with the first variable and end with the last one, without any explanation or markdown.",
        "meta/main.yml": "Generate the metadata YAML code for the Ansible role for {tool} from the repository {repo}. Start with the first metadata item and end with the last one, without any explanation or markdown.",
    }

    for tool_data in csv_data:
        tool_name = tool_data['Asset']
        repo = tool_data['Repo']

        for filename in role_structure:
            path = f"{tool_name}/{filename}"
            try:
                if filename.endswith('/'):
                    # This is a directory, create it
                    os.makedirs(path, exist_ok=True)
                else:
                    # This is a file, create it and write the content
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, 'w') as file:
                        if filename in prompts:
                            prompt = prompts[filename].format(tool=tool_name, repo=repo)
                            response = openai.ChatCompletion.create(
                                model="gpt-4",
                                messages=[
                                    {"role": "system", "content": "You are a helpful assistant."},
                                    {"role": "user", "content": prompt}
                                ]
                            )
                            content = response['choices'][0]['message']['content']
                            # remove any leading or trailing whitespace
                            content = content.strip()
                            file.write(content)
                        else:
                            file.write("# TODO: Implement this file\n")
            except Exception as e:
                logging.error(f"Failed to create file or directory {path}", exc_info=True)
                return jsonify({"error": f"Failed to create file or directory {path}"}), 500

    return jsonify({"message": "Directory structure and files created"}), 200

if __name__ == "__main__":
    app.run(debug=True)
