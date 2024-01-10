import json
import os
import openai
import pathlib
from dotenv import load_dotenv

load_dotenv()

class GPTInterpreter:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key

    def interpret(self, user_data: str) -> str:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_data}
                ]
            )
            if response.choices and response.choices[0].message:
                return response.choices[0].message['content']
            
        except Exception as e:
            print(f"Exception caught while trying to interpret user data: {e}")

        return None

def generate_tree_structure(path=pathlib.Path(__file__).parent.absolute(), level=0):
    result = ""
    for item in path.iterdir():
        indent = " " * (level * 4) + "|-- "
        if item.is_file() or item.name == '.env':
            rel_path = str(item.resolve().relative_to(path))
            result += indent + rel_path.replace("\\", "/") + "\n"
        elif not "__pycache__" in str(item):
            result += indent + str(item.name) + "\n"
            result += generate_tree_structure(item, level + 1)
    return result

def update_source_code_map():
    tree_output = generate_tree_structure()
    handbook_file_path = os.path.join(os.path.dirname(__file__), 'handbook.json')

    if not os.path.isfile(handbook_file_path):
        with open(handbook_file_path, "w") as file:
            file.write('{}') 

    try:
        with open(handbook_file_path, "r") as file:
             data = json.load(file)
    except Exception as e:
            print(f"Exception caught while trying to load JSON file: {e}")
            return None

    data["source_code_map"] = tree_output

    try:
        with open(handbook_file_path, 'w') as file:
            json.dump(data, file, indent=2)
    except Exception as e:
        print(f"Exception caught while trying to dump JSON data: {e}")  

class SelfImprove:
    def __init__(self, gpt_interpreter: GPTInterpreter):
        self.gpt_interpreter = gpt_interpreter

    def map_source_code(self):
        update_source_code_map()
        print("\nSource code map has been updated.")

    def analyze_source_code(self, input_value: str):
        if input_value.lower() == 's':
            source_files = []
            for root, dirs, files in os.walk("."):
                if root == "./__pycache__":
                    continue
                for file in files:
                    if file.endswith(".py") and file != "source-code_labeled.json" and file != "handbook.json":
                        source_files.append(os.path.join(root, file))
            total_files = len(source_files)
            source_data = {}

            base_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(base_dir, "source-code_labeled.json")
            with open(file_path, "w") as f:
                json.dump(source_data, f, indent=2)           
                for index, file in enumerate(source_files, start=1):
                    with open(file, "r") as reader:
                        content = reader.read()

                    print(f"Analyzing file {file} ({index}/{total_files})")
                    summary = self.gpt_interpreter.interpret("Please provide a brief summary of the contents of the Python file '{file}' with content:\n{content}")

                    source_data[file] = {
                        "file_title": os.path.basename(file),
                        "file_path": file,
                        "file_code": content,
                        "file_summary": summary
                    }

                json.dump(source_data, f, indent=2)

            print("\nSource code analysis has been completed.")
        else:
            print("\nUsing the existing 'source-code_labeled.json' file.")

    def get_self_improvement_instructions(self) -> str:
        instructions = input("\nPlease provide instructions for Thalis' self-improvement: ")
        return instructions

    def find_relevant_files(self, instructions: str):
        data = None
        base_dir = os.path.dirname(os.path.abspath(__file__))
        json_file_path = os.path.join(base_dir, "source-code_labeled.json")

        with open(json_file_path, "r") as f:
            data = json.load(f)

        relevant_files = []
        for key, value in data.items():
            relevance = self.gpt_interpreter.interpret({
                "action": "evaluate_relevance",
                "file_summary": value["file_summary"],
                "instructions": instructions,
            })
            if relevance and relevance.lower() == "true": 
                relevant_files.append(value["file_path"])

        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, "handbook.json")
        
        with open(file_path, "r+") as file:
            handbook_data = json.load(file)
            handbook_data["relevant_files"] = relevant_files
            file.seek(0)      
            json.dump(handbook_data, file, indent=2)

        print("\nRelevant files have been identified. Here is the list of the files:")
        for file in relevant_files:
            print(f"- {file}")
        print("\nUpdated the 'relevant_files' section in the 'handbook.json' file.")

    def improve(self):
        self.map_source_code()
        strategy = input("\nWould you like to analyze the source code again or use the existing 'source-code_labeled.json'? (Answer with 's' for scan and 'e' for existing): ")
        self.analyze_source_code(strategy)
        instructions = self.get_self_improvement_instructions()
        self.find_relevant_files(instructions)

if __name__ == "__main__":
    gpt_interpreter_instance = GPTInterpreter()
    self_improvement_instance = SelfImprove(gpt_interpreter=gpt_interpreter_instance)
    self_improvement_instance.improve()
