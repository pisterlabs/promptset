# %%
import logging
import os
import re
from queue import Queue
from dotenv import load_dotenv
import openai
import json
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Load environment variables from .env file
load_dotenv()

# Access the API key from the environment variable
openai_api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = openai_api_key
# Set up logging
logging.basicConfig(level=logging.INFO)



import json
import openai
from code_search import similarity_search

query = "Variable impedance control for force feedback"
results_string = similarity_search(query)
print(results_string)


completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-16k-0613",
    messages=[
        {
            "role": "system",
            "content": "You are an AI that decomposes complex code generation tasks into smaller, manageable sub-tasks. Each sub-task should be a independent file, should contain the name of the python file, and should contain the name of the file,description, all the functions and classes from the file, as well releted files."
        },
        {
            "role": "user",
            "content": f"Given the complex code generation task: 'Write a variable impedance control for force feedback using ros2, webots, webots_ros2 and ros2_control.', please decompose it into a detailed, numbered list of sub-tasks. Each sub-task should be a independent file should contain the name of the python file, description,all the functions and classes from the file, as well releted files. Make sure to devide the task into minimum 5 files. Try to make the code as readable as possible, encapsulating the code into functions and classes. Lets think step by step.\n\nThe following are the retrived documents from all the relevant repositories based on the query 'Variable impedance control for force feedback':\n{results_string}\nThese retrived functions from all the relevant repositories are helpfull but not fullfile the user task. Please use this context to help guide the task decompositionThese retrived functions from all the relevant repositories are helpfull but not fullfile the user task. Please use this context to help guide the task decomposition"
        }
    ],
    functions=[
        {
            "name": "generate_code",
            "description": "Generates the code for multiple files, each described by a dictionary of attributes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "description": "An array of dictionaries, each representing a file. Each dictionary should include 'order' (the order of development), 'code_blocks' (an array of dictionaries detailing the code blocks in the file).",
                        "items": {
                            "type": "object",
                            "properties": {
                                "order": {
                                    "type": "integer",
                                    "description": "The order of development for the file."
                                },
                                "code_blocks": {
                                    "type": "array",
                                    "description": "An array of dictionaries, each detailing a code block in the file. Each dictionary should include 'type' (either 'function' or 'class'), 'name' (the name of the function or class), 'description' (a description of the block's purpose), 'content' (the details of the function or class, including function arguments or class methods, as applicable), and 'related_files' (an array of filenames that are related to the code block).",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "type": {
                                                "type": "string",
                                                "description": "The type of the code block, either 'function' or 'class'."
                                            },
                                            "name": {
                                                "type": "string",
                                                "description": "The name of the function or class."
                                            },
                                            "description": {
                                                "type": "string",
                                                "description": "A description of the block's purpose."
                                            },
                                            "content": {
                                                "type": "string",
                                                "description": "The details of the function or class, including arguments and methods as applicable."
                                            },
                                           
                                            }
                                        },
                                        "required": ["type", "name", "description", "content"]
                                    }
                                }
                            },
                            "required": ["order", "code_blocks"]
                        }
                    }
                },
"required": ["files"]
            }
        
    ],
    function_call={"name": "generate_code"}
)

reply_content = completion.choices[0]
print(reply_content)

args = reply_content["message"]['function_call']['arguments']
data = json.loads(args)

# Initialize an empty dictionary to store the files
files = {}

# Go through each file
for file in data["files"]:
    # Create a new dictionary for this file
    files[file["code_blocks"][0]["name"]] = {
        "order": file["order"],
        "code_blocks": file["code_blocks"],
    }

# Sort the files dictionary based on the order of development
files = dict(sorted(files.items(), key=lambda item: item[1]['order']))

# Print the files dictionary
for filename, file_data in files.items():
    print(f"Order of development: {file_data['order']}")
    print(f"{filename}:")
    for block in file_data['code_blocks']:
        print(f"  Code block type: {block['type']}")
        print(f"  Code block name: {block['name']}")
        print(f"  Code block description: {block['description']}")
        print(f"  Code block content: {block['content']}")
        #print(f"  Related files: {block['related_files']}")


files_string = ""
for filename, file_data in files.items():
    files_string += f"Order of development: {file_data['order']}\n"
    files_string += f"{filename}:\n"
    for block in file_data['code_blocks']:
        files_string += f"  Code block type: {block['type']}\n"
        files_string += f"  Code block name: {block['name']}\n"
        files_string += f"  Code block description: {block['description']}\n"
        files_string += f"  Code block content: {block['content']}\n"
        #files_string += f"  Related files: {block['related_files']}\n"

# %%

completion2 = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-16k-0613",
    messages=[
        {
            "role": "system",
            "content": "You are an advanced AI with capabilities to analyze intricate code and pseudocode files. Based on this analysis, you provide recommendations for the most appropriate vectorstore repositories to extract relevant code snippets from. In addition, you generate search queries that could be utilized to fetch these helpful code samples."
        },
        {
            "role": "user",
            "content": f"I need your expertise to examine the provided code and pseudocode files. Your task is to pinpoint any issues, inefficiencies, and areas for potential enhancements. Here are the files you need to look into:\n\n{files_string}"
        }
    ],
functions=[
    {
        "name": "analyze_code",
        "description": "This function performs an analysis on the provided code files. It returns a list of suitable repositories for fetching relevant code samples and suggests a respective search query for each repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file_name": {
                                "type": "string",
                                "description": "The name of the code file."
                            },
                            "repository": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "The name of the repository.",
                                        "enum": ['db_ros2_control', 'db_ros2', 'db_webots_ros2', 'db_webots']
                                    },
                                    "query": {
                                        "type": "string",
                                        "description": "The search query designed to fetch code samples from the specified repository."
                                    }
                                },
                                "required": ["name", "query"],
                                "description": "An object representing a repository and a corresponding search query."
                            }
                        },
                        "required": ["file_name", "repository"]
                    },
                    "description": "An array of objects, each representing a code file that needs to be analyzed."
                }
            },
            "required": ["files"]
        }
    }
],
function_call={"name": "analyze_code"}


    )
    


reply_content2 = completion2.choices[0]
print(reply_content2)

args2 = reply_content2["message"]['function_call']['arguments']
data2 = json.loads(args2)


print(data2)


# Define the list of directories to search in
directories = ['db_ros2_control', 'db_ros2', 'db_webots_ros2', 'db_webots']

# Create an empty dictionary to store the results
results = {}

# Loop through each file in the data2 dictionary
for file_data in data2["files"]:
    file_name = file_data["file_name"]
    
    repository = file_data["repository"]
    query = repository['query']
    
    # Call the similarity_search function and save the result as a string
    result = similarity_search(query, directories)
    
    # Store the result in the dictionary, using the filename_query as the key
    results[f"{file_name}_{query}"] = result

# Create a dictionary to store the strings for each file
file_strings = {}

# Loop through each file in the files dictionary
for filename, file_data in files.items():
    # Create a list to store the lines for this file
    file_lines = []
    file_lines.append(f"Order of development: {file_data['order']}")
    file_lines.append(f"{filename}:")
    
    for block in file_data['code_blocks']:
        file_lines.append(f"  Code block type: {block['type']}")
        file_lines.append(f"  Code block name: {block['name']}")
        file_lines.append(f"  Code block description: {block['description']}")
        file_lines.append(f"  Code block content: {block['content']}")
        
        # Loop through the results dictionary to find the results for this file
        for key, value in results.items():
            # If the filename is in the key of the results dictionary, add the query and its result to the lines
            if filename in key:
                file_lines.append(f"#####################################")
                file_lines.append(f"  Query:\n\n {key.replace(filename+'_', '')}")
                file_lines.append(f"\n\n  Query result:\n\n {value}")
    
    # Join the lines for this file into a single string and add it to the file_strings dictionary
    file_strings[filename] = '\n'.join(file_lines)


# %%
print(file_strings["impedance_control.py"])

# %%
# Loop through each file_string in the file_strings dictionary
for filename, file_string in file_strings.items():
    print(f"File: {filename}")
    print(file_string)

# %%
# Loop through each file_string in the file_strings dictionary
for filename, file_string in file_strings.items():
    print(f"File: {filename}")
    print(file_string)

# %%
# Define a string to store the print outputs
output_string = ""

# Loop through each file_string in the file_strings dictionary
for filename, file_string in file_strings.items():
    # Create a new completion with the file_string as the user message content
    completion4 = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {
                "role": "system",
                "content": "You are an AI Code Optimization Model that can optimize code, complete #TODOs, recommend best practices, and learn from relevant code repositories. Your main task is to analyze the given code, which performs semantic search queries on a vectorstore using OpenAI embeddings, and apply the insights from the search results to refine the code. The final goal is to produce a fully functional and optimized code file that can be used as part of a larger project, specifically a variable impedance control code for force feedback"
            },
            {
                "role": "user",
                "content": f"I am working on a coding project that aims to develop a variable impedance control code for force feedback. I need your expertise to improve my code. Here is the current version of one file of my code, along with the semantic search queries I’ve done on a vectorstore using OpenAI embeddings, and the results of these queries:\n\n{file_string}\n\nCan you improve this code, using the suggestions from the semantic search results? Please write the improved and complete code file. Please complete and improve the file based on the context."
            }
        ],
    )

    new_files = {}
    new_files[filename] = completion4.choices[0].message['content']
    # Append to the output_string instead of printing
    output_string += f"For file: {filename}, the improved code is: {new_files[filename]}\n"

# Now you can print or further process the output_string as required
print(output_string)
    # Print or process the completion as needed
    #print(f"For file: {filename}, the improved code is: {new_files[filename]}\n")


# %%
print(output_string)

# %%
completion4 = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-16k-0613",
    messages=[
        {
            "role": "system",
            "content": "You are a highly skilled AI, capable of analyzing complex code and pseudocode documents. Your purpose is to provide insights on potential improvements, identify inefficiencies and issues, and recommend the most relevant repositories from VectorStore to retrieve suitable code snippets. Additionally, you create specific search queries to access these valuable code samples."
        },
        {
            "role": "user",
            "content": f"I require your expertise to examine the following code and pseudocode documents. Identify potential problems, inefficiencies, and opportunities for enhancement. You final goal is to rewrite and finish the code to make full functional. Here are the documents for your review:\n\n{files_string}"
        }
    ],
    functions=[
    {
        "name": "search_code",
        "description": "This function utilizes VectorStore's semantic search for code functions. It receives the name of a repository and a corresponding query, along with the name of a code document, and returns the code snippets from the respective repository that are most similar to the queries.",
        "parameters": {
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file_name": {
                                "type": "string",
                                "description": "The name of the code document."
                            },
                            "repositories": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "description": "The name of the repository.",
                                            "enum": ['db_ros2_control', 'db_ros2', 'db_webots_ros2', 'db_webots']
                                        },
                                        "query": {
                                            "type": "string",
                                            "description": "The search query tailored to retrieve code snippets from the specified repository."
                                        }
                                    },
                                    "required": ["name", "query"]
                                },
                                "description": "An array of objects, each one representing a repository and a corresponding search query."
                            }
                        },
                        "required": ["file_name", "repositories"]
                    },
                    "description": "An array of objects, each one representing a code document that needs to be reviewed."
                }
            },
            "required": ["files"]
        }
    }
],
function_call={"name": "search_code"}
)


# %%
import openai
from termcolor import colored
from code_search import similarity_search
def pretty_print_conversation(messages):
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "function": "magenta",
    }
    formatted_messages = []
    for message in messages:
        if message["role"] == "system":
            formatted_messages.append(f"system: {message['content']}\n")
        elif message["role"] == "user":
            formatted_messages.append(f"user: {message['content']}\n")
        elif message["role"] == "assistant" and message.get("function_call"):
            formatted_messages.append(f"assistant: {message['function_call']}\n")
        elif message["role"] == "assistant" and not message.get("function_call"):
            formatted_messages.append(f"assistant: {message['content']}\n")
        elif message["role"] == "function":
            formatted_messages.append(f"function ({message['name']}): {message['content']}\n")
    for formatted_message in formatted_messages:
        print(
            colored(
                formatted_message,
                role_to_color[messages[formatted_messages.index(formatted_message)]["role"]],
            )
        )

def search_code_completion_request(messages, functions):
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        messages=messages,
        functions=functions,
        function_call={"auto"}
    )


functions = [
    {
        "name": "similarity_search",
        "description": "Vectorstore embedding semantic search for code functions. It receives a query and the directories to search, and returns the most similar code snippets to the queries.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query designed to fetch code samples."
                },
                "directories": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["db_webots", "db_ros2", "db_webots_ros2", "db_ros2_control"],
                        "description": "The directories in which to perform the search."
                    },
                    "description": "An array of directory names."
                }
            },
            "required": ["query", "directories"]
        }
    },
]



messages = [
    {
        "role": "system",
        "content": "You are a sophisticated AI that has the ability to analyze complex code and pseudocode documents. You are tasked with making necessary clarifications in a series of chat turns until you gather sufficient information to rewrite the code. You can utilize the 'search_code' function to fetch relevant code snippets based on semantic similarity, and subsequently improve the given file."
    },
    {
        "role": "user",
        "content": f"I need your assistance in reviewing these code and pseudocode documents. You final goal is to rewrite and finish the code to make full functional The final goal is to create a project for variable impedance control providing force feedback. The project will use Webots, ROS2, webots_ros2, and ros2_control. You are required to identify potential problems, inefficiencies, and areas for improvements in these documents. Here are the documents you need to work on:\n\n{output_string}\n\nPlease first clarify any question that you need to finish the code with me. After you completely understand the goal of the user, use the search_code function to find relevant code that can help improve the code."  
    }
]

while True:
    chat_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        messages=messages,
        functions=functions,
    )
    assistant_message = chat_response['choices'][0]['message']
    messages.append(assistant_message)

    pretty_print_conversation(messages)
    
    if assistant_message.get("function_call"):
        # Extract the function name and arguments from the message
        function_name = assistant_message["function_call"]["name"]
        arguments = json.loads(assistant_message["function_call"]["arguments"])

        print(f"Function Name: {function_name}")
        print(f"Arguments: {arguments}")

        # Call the function and get the results
        if function_name == "similarity_search":
            # Initialize a results list
            all_results = []

            
            print("Calling similarity_search function...")
            results = similarity_search(arguments['query'], arguments['directories'])
            print("similarity_search function returned.")

            print(f"Results: {results}")



