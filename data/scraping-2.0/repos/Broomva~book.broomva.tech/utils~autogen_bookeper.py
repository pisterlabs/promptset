import glob
import os

import openai
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json

# Load configuration from JSON or environment variable
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to parse folder and read file contents
def parse_folder(folder_path):
    parsed_info = {}
    for filepath in glob.glob(os.path.join(folder_path, '**'), recursive=True):
        if os.path.isfile(filepath):
            with open(filepath, 'r') as file:
                file_content = file.read()
                parsed_info[filepath] = file_content
    return parsed_info

# Function to build a markdown file from parsed information
def build_md(parsed_info, file_path):
    with open(file_path, 'w') as file:
        for filepath, content in parsed_info.items():
            file.write(f'## {filepath}\n\n')
            file.write(f'{content}\n\n')

llm_config_parser = {
    "functions": [
        {
            "name": "parse_folder",
            "description": "Parse information from a local folder",
            "parameters": {
                "folder_path": {
                    "type": "string",
                    "description": "The path to the folder to parse"
                }
            },
            "schema": {  # Corrected schema definition
                "type": "object",
                "properties": {
                    "folder_path": {
                        "type": "string"
                    }
                },
                "required": ["folder_path"]
            }
        },
        {
            "name": "build_md",
            "description": "Build a .md file from parsed information",
            "parameters": {
                "parsed_info": {
                    "type": "dict",
                    "description": "The parsed information"
                },
                "file_path": {
                    "type": "string",
                    "description": "The path to the .md file to write"
                }
            },
            "schema": {
                "type": "object",
                "properties": {
                    "parsed_info": {
                        "type": "dict"
                    },
                    "file_path": {
                        "type": "string"
                    }
                },
                "required": ["parsed_info", "file_path"]
            }
        }
    ],
    "config_list": config_list
}


# Create instances of AssistantAgent and UserProxyAgent
parser = AssistantAgent(
    name="parser",
    system_message="Parse information from local folders and build consolidated .md files with utmost detail about each topic linking knowledge from all files within the folder",
    llm_config=llm_config_parser,
)

user_proxy = UserProxyAgent(
    name="User_proxy",
    code_execution_config={"last_n_messages": 2, "work_dir": "coding"},
    is_termination_msg=lambda x: x.get("content", "") and x.get(
        "content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="TERMINATE",
    function_map={
        "parse_folder": parse_folder,
        "build_md": build_md,
    }
)

# Initiate chat with the parser agent
user_proxy.initiate_chat(parser, message='lets start by reviewing the folders we have access to')
