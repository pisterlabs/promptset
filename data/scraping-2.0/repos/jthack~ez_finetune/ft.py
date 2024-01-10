import openai
from packaging import version

required_version = version.parse("1.1.1")
current_version = version.parse(openai.__version__)

if current_version < required_version:
    raise ValueError(f"Error: OpenAI version {openai.__version__}"
                     " is less than the required version 1.1.1")
else:
    print("OpenAI version is compatible.")

import os
import json from openai import OpenAI

class FineTuningDataGenerator:
    def __init__(self, persona, folder_path):
        self.persona = persona
        self.folder_path = folder_path
        self.client = OpenAI()

    def generate_query(self, file_content):
        response = self.client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "You are a query generator. For given user content, you will write a query that might have resulted in the content as output. For example, if the user's content is a poem about sharks. You would output 'Write a poem about sharks'."},
                {"role": "user", "content": file_content}
            ]
        )
        return response.choices[0].message.content.strip()


    def process_file(self, file_path):
        with open(file_path, 'r') as file:
            file_content = file.read()
        query = self.generate_query(file_content)
        return {
            "messages": [
                {"role": "system", "content": self.persona},
                {"role": "user", "content": query},
                {"role": "assistant", "content": file_content}
            ]
        }

    def generate_data(self):
        training_data = []
        for file_name in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, file_name)
            if os.path.isfile(file_path):
                data = self.process_file(file_path)
                training_data.append(data)
        return training_data
def main(persona, folder_path, output_file):
    generator = FineTuningDataGenerator(persona, folder_path)
    training_data = generator.generate_data()
    with open(output_file, 'w') as f:
        for data in training_data:
            f.write(json.dumps(data) + '\n')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate fine-tuning data for GPT models.")
    parser.add_argument("--persona", required=True, help="The fine-tuning persona/system prompt.")
    parser.add_argument("--folder", required=True, help="The folder containing documents for fine-tuning.")
    parser.add_argument("--output", required=True, help="The file to write the training data to.")
    args = parser.parse_args()
    main(args.persona, args.folder, args.output)

