import os
import requests
import json
import openai
from openai import OpenAI
import logging

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app.log',
    filemode='a'
)




def get_client():
    client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key= os.environ.get("OPENAI_API_KEY")

    )
    
    return client

def send_text_request(prompt, system_prompt="You are a helpful assistant.", context=""):
    client = get_client()

    response = client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt}])
    return response

def send_json_request(prompt, system_prompt="You are a helpful assistant. Please respond in JSON", context=""):
    client = get_client()
    response = client.chat.completions.create(
    model="gpt-4-1106-preview",
    response_format={"type" : "json_object"},
    messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt}])
    return response

def generate_mvp(description):
    """
    Generates a minimum viable product (MVP) from the specified description.
    :param description: str, the description of the task.
    :return: str, the generated MVP.
    """
    system_prompt = "You are a product expert with deep knowledge in software development. "
    system_prompt += "Given the following description of a software product, create a detailed "
    system_prompt += "minimum viable product (MVP) description. The MVP must be a Python program. "
    system_prompt += "Please outline the core functionalities, key features, and basic architecture. "
    system_prompt += "Include considerations for scalability, target user demographics, "
    system_prompt += "and any preferred technologies or frameworks. Ensure the MVP description "
    system_prompt += "is detailed and practical for implementation. Include the user's description "
    system_prompt += "at the beginning of your description."

    response = send_text_request(description, system_prompt)
    return parse_text_response(response)

def synthesize_best_mvp(mvp1, mvp2, mvp3):
    """
    Analyzes multiple MVP descriptions and synthesizes a refined MVP using the best ideas from each.

    Args:
        mvp_list: list of str, a list containing multiple MVP descriptions.

    Returns:
        str: The synthesized MVP description.
    """
    combined_mvp_descriptions =f"{mvp1}\n\n{mvp2}\n\n{mvp3}"
    system_prompt = "You are a highly skilled product manager. "
    system_prompt += "Given the following three MVP descriptions, synthesize a refined MVP. "
    system_prompt += "Extract the best elements from each description and combine them into a single, optimized MVP. "
    system_prompt += "Eliminate any redundant or unnecessary elements. "
    system_prompt += "Ensure the final MVP is coherent, practical, and captures the essence of the best ideas. "
    prompt = f"MVP Descriptions:\n\n{combined_mvp_descriptions}"

    response = send_text_request(prompt, system_prompt)
    return parse_text_response(response)

def synthesize_best_code(component_description, file_name, mvp_description, file_code1, file_code2, file_code3):
    """
    Synthesizes the best aspects of three code attempts into a refined version of a file.

    Args:
        component_description: str, the description of the component.
        file_name: str, the name of the file to be synthesized.
        mvp_description: str, the description of the MVP.
        file_code1: str, the first attempt at writing the file.
        file_code2: str, the second attempt.
        file_code3: str, the third attempt.
    
    Returns:
        str: The synthesized and improved code for the file.
    """
    system_prompt = "You are a highly skilled software engineer and code synthesizer. "
    system_prompt += "Given three different attempts at coding a file, synthesize the best parts from each to create an optimized version of the file. "
    system_prompt += "Focus on ensuring that all functions are fully implemented with valid code, and all necessary dependencies are referenced. "
    system_prompt += "The final code should align with the provided component description and the overall MVP context. "
    prompt = f"File Name: {file_name}\n"
    prompt += f"Component Description: {component_description}\n"
    prompt += f"MVP Description: {mvp_description}\n\n"
    prompt += "Code Attempt 1:\n" + file_code1 + "\n\n"
    prompt += "Code Attempt 2:\n" + file_code2 + "\n\n"
    prompt += "Code Attempt 3:\n" + file_code3

    response = send_text_request(prompt, system_prompt)
    txt_reponse = parse_text_response(response)
    return extract_code_from_string(txt_reponse)

def get_component_list(mvp_description):
    """
    Generates a list of components from the specified MVP description.
    :param mvp_description: str, the MVP description.
    :return: list, the generated component list.
    """
    with open("comp_list_template.txt", 'r') as f:
        comp_list_template = f.read()
    system_prompt = "You are an experienced software architect with expertise in creating Minimal Viable Products (MVPs) using Python. "
    system_prompt += "Given the provided MVP description, your task is to identify and list only the essential components required to build the MVP. "
    system_prompt += "Focus on simplicity and necessity, avoiding any superfluous features or components that do not directly contribute to the MVP's core functionality. "
    system_prompt += "For each identified component, briefly outline the corresponding essential files, primary functions, core data structures, and critical dependencies. "
    system_prompt += "Organize this information in a concise JSON format. "
    system_prompt += "The architecture should be straightforward, promoting modularity and maintainability, while ensuring that each component is vital for the MVP's operation. "
    system_prompt += "Remember, the goal is to create a streamlined, efficient architecture that embodies the MVP philosophy of 'less is more.'"
    system_prompt += "make sure the components are in the following format: \n"
    system_prompt += f"{comp_list_template}\n"
    response = send_json_request(mvp_description, system_prompt)
    return parse_json_response(response)


def parse_text_response(response):
    """
    Parses the response received from the LLM API.
    :param response: dict, the response from the API.
    :return: str, the text generated by the LLM.
    """
    #print(response.choices[0].message.content)
    logging.info(response.choices[0].message.content)
    return response.choices[0].message.content

def parse_json_response(response):
    """
    Parses the response received from the LLM API.
    :param response: dict, the response from the API.
    :return: list, the JSON object generated by the LLM.
    """
    logging.info(response.choices[0].message.content)
    return json.loads(response.choices[0].message.content)

def handle_api_error(response):
    """
    Manages errors encountered during API interaction.
    :param response: dict, the response from the API.
    :return: str, error message if any, else an empty string.
    """
    if response.status_code != 200:
        return f"API Error: {response.status_code} - {response.text}"
    return ""


def generate_py_file_for_component(component, file_name, mvp_context):
    """
    Generates a file for a specific file for a component based on the MVP context, 
    considering the file extension to determine the programming language.
    
    Args:
        component: str, the name of the component.
        file_name: str, the name of the file to be generated, including its extension.
        mvp_context: str, the context or description of the MVP.
    
    Returns:
        str: The generated code for the component file, appropriate to its file type.
    """
    file_extension = file_name.split('.')[-1]  # Extract file extension
    language = determine_language_from_extension(file_extension)  # Function to determine the language from the extension

    system_prompt = "You are a multi-language software developer. "
    system_prompt += "Given a component name, a file name with its extension, and the context of a Minimum Viable Product (MVP), "
    system_prompt += "your task is to create a file that performs the tasks specified for the component. "
    system_prompt += "Use the file extension to determine the programming language or markup language. "
    system_prompt += "Ensure the output contains only valid code or markup and necessary comments, "
    system_prompt += "and that it aligns with the MVP context provided. "
    prompt = f"Component Name: {component}. "
    prompt += f"MVP Context: {mvp_context}. "
    prompt += f"File Name and Extension: {file_name}. "
    prompt += f"Language: {language}. "
    prompt += "Please generate content that accurately reflects the specified file within the component within the MVP framework."

    response = send_text_request(prompt, system_prompt)
    txt_reponse = parse_text_response(response)
    return extract_code_from_string(txt_reponse)

def extract_code_from_string(input_string):
    """
    Extracts the code enclosed between specific markers in a string.

    Args:
        input_string: str, the input string from which code needs to be extracted.

    Returns:
        str: Extracted code.
    """
    start_markers = []
    start_markers.append("```python")
    start_markers.append("```html")
    start_markers.append("```js")
    start_markers.append("```javascript")
    start_markers.append("```css")
    start_markers.append("```java")
    start_markers.append("```cpp")
    start_markers.append("```Dockerfile")
    start_markers.append("```yaml")
    start_markers.append("```nginx")


    end_marker = "```"
    code = []
    capture = False

    inputarr = input_string.split('\n')

    for line in input_string.split('\n'):
        if not capture:
            for start_marker in start_markers:
                if start_marker in line:
                    capture = True
            continue

        if end_marker in line and capture:
            break
        if capture:
            code.append(line)

    return '\n'.join(code)



def determine_language_from_extension(extension):
    """
    Determines the programming or markup language from the file extension.

    Args:
        extension: str, the file extension.

    Returns:
        str: The name of the language associated with the extension.
    """
    # Mapping of file extensions to languages
    extension_to_language = {
        'py': 'Python',
        'html': 'HTML',
        'js': 'JavaScript',
        'css': 'CSS',
        'java': 'Java',
        'cpp': 'C++',
        'c': 'C',
        'cs': 'C#',
        'php': 'PHP',
        'rb': 'Ruby',
        'swift': 'Swift',
        'kt': 'Kotlin',
        'ts': 'TypeScript',
        'jsx': 'JSX',
        'go': 'Go',
        'rs': 'Rust',
        'sql': 'SQL',
        'sh': 'Shell Script',
        'bat': 'Batch Script',
        'pl': 'Perl',
        'r': 'R',
        'xml': 'XML',
        'json': 'JSON',
        'yaml': 'YAML',
        'yml': 'YAML',
        'md': 'Markdown',
        'txt': 'Text',
        # Add more mappings as needed
    }
    return extension_to_language.get(extension.lower(), 'Unknown')


def document_code_file(file_name, component_description, mvp_context):
    """
    Generates documentation for a Python file, considering the component description and MVP context.

    Args:
        file_name: str, the name of the file to be documented.
        component_description: str, a description of the component.
        mvp_context: str, the context or description of the MVP.
    
    Returns:
        str: The generated documentation for the file.
    """
    with open(file_name, 'r') as f:
        file_content = f.read()
    system_prompt = "You are an expert in Python programming and technical writing. "
    system_prompt += "Given the content of a Python file, component description, and MVP context, "
    system_prompt += "create detailed documentation for the file. "
    system_prompt += "The documentation should describe the purpose and functionality of each function and class in the file, "
    system_prompt += "and how they relate to the component and overall MVP. "
    system_prompt += "Please ensure the documentation is clear, concise, and useful for understanding the file's role in the project.\n"
    prompt = f"Component Description: {component_description}\n"
    prompt += f"MVP Context: {mvp_context}\n"
    prompt += f"Python File Content:\n\n{file_content}"

    response = send_text_request(prompt, system_prompt)
    return parse_text_response(response)

def critique_code_file(file_name, component_description, mvp_context):
    """
    Critiques a Python file, providing insights on improvements, missing elements, 
    and alignment with the component description and MVP context.

    Args:
        file_content: str, the content of the Python file.
        file_name: str, the name of the file to be critiqued.
        component_description: str, a description of the component.
        mvp_context: str, the context or description of the MVP.
    
    Returns:
        str: The critique of the Python file.
    """
    with open(file_name, 'r') as f:
        file_content = f.read()
    system_prompt = "You are an experienced software developer and code reviewer. "
    system_prompt += "Your task is to analyze a Python file in the context of its intended component and MVP. "
    system_prompt += "Focus on identifying any missing or obviously incorrect functionalities, "
    system_prompt += "and check for missing or incorrect dependencies. "
    system_prompt += "Offer concise suggestions for resolving these specific issues. "
    prompt = f"Component Description: {component_description}\n"
    prompt += f"MVP Context: {mvp_context}\n"
    prompt += f"Python File Content:\n\n{file_content}"

    response = send_text_request(prompt, system_prompt)
    return parse_text_response(response)

def generate_unit_tests_for_file(file_name, component_description, mvp_context):
    """
    Generates comprehensive unit tests for a Python file based on its content, 
    component description, and MVP context.

    Args:
        file_content: str, the content of the Python file.
        file_name: str, the name of the file for which unit tests are to be generated.
        component_description: str, a description of the component.
        mvp_context: str, the context or description of the MVP.
    
    Returns:
        str: The generated unit tests for the Python file.
    """
    with open(file_name, 'r') as f:
        file_content = f.read()
    system_prompt = "You are an expert in Python development with a focus on software testing. "
    system_prompt += "Given the content of a Python file, along with its component description and MVP context, "
    system_prompt += "your task is to create a comprehensive set of unit tests for the file. "
    system_prompt += "These tests should comprehensively cover all functions and classes, "
    system_prompt += "validating correctness, handling edge cases, and ensuring alignment with the component's role in the MVP. "
    system_prompt += "The tests should provide clear pass/fail output and be ready for immediate use. "
    prompt = f"Component Description: {component_description}\n"
    prompt += f"MVP Context: {mvp_context}\n"
    prompt += f"Python File Content:\n\n{file_content}"

    response = send_text_request(prompt, system_prompt)
    txt_reponse = parse_text_response(response)
    return extract_code_from_string(txt_reponse)

def auto_fix_test_failure(file_content, unit_test_content, unit_test_output):
    """
    Analyzes unit test output, and if a failure is detected, attempts to modify the original file to fix the problem.

    Args:
        file_content: str, the content of the original Python file.
        unit_test_content: str, the content of the unit test.
        unit_test_output: str, the output from running the unit test.

    Returns:
        str: The modified version of the original file, or None if no fix could be suggested.
    """
    # Check if the unit test output indicates a failure
    if 'FAILED' in unit_test_output:
        # Create a prompt for the AI to suggest modifications
        system_prompt = "As an expert Python developer, analyze the following unit test output, "
        system_prompt += "along with the content of the original file and the unit test. "
        system_prompt += "Suggest modifications to the original file to fix the failures identified in the unit test. "
        system_prompt += f"Original File Content:\n\n{file_content}\n\n"
        system_prompt += f"Unit Test Content:\n\n{unit_test_content}\n\n"
        system_prompt += f"Unit Test Output:\n\n{unit_test_output}\n"
        system_prompt += "Suggested Modifications:"

        # Send the prompt to the AI and get a response
        response = send_text_request(system_prompt)

        # Extract the suggested modifications
        suggested_modifications = parse_text_response(response)

        # Apply the modifications to the original file content
        # This step is highly contextual and would depend on the nature of the modifications suggested
        # For simplicity, let's assume the AI provides a modified file content directly
        modified_file_content = suggested_modifications

        return modified_file_content
    else:
        # No failure detected, return None
        return None

def generate_main_entry_point(mvp_description, component_list):
    """
    Generates the main.py file, which is the entry point of a program, based on the MVP description and component list.

    Args:
        mvp_description: str, the description of the MVP.
        component_list: list, a list of components that need to be integrated into the main.py file.

    Returns:
        str: The content of the generated main.py file.
    """
    # Format the component list into a readable string for the prompt
    formatted_component_list = ', '.join(component_list)

    # Create a prompt for the AI to generate the main.py content
    system_prompt = "As an experienced software engineer, create the main.py file for a Python program. "
    system_prompt += "This file should serve as the entry point of the program. "
    system_prompt += f"Consider the following MVP description: {mvp_description}. "
    system_prompt += f"The program must integrate these components: {formatted_component_list}. "
    system_prompt += "The main.py file should initialize and orchestrate the components effectively, "
    system_prompt += "ensuring they work together as described in the MVP. "
    system_prompt += "Include necessary imports, initialization, and execution flow. The code should be clean, efficient, and well-documented."

    # Send the prompt to the AI and get a response
    response = send_text_request(system_prompt)

    # Extract the generated main.py content
    main_py_content = parse_text_response(response)

    return main_py_content



# This can be used for testing the functions in this file.
if __name__ == "__main__":
    # Example usage and testing
    test_prompt = "Translate the following English text to French: 'Hello, world!'"
    response = send_text_request(test_prompt)
    print(parse_text_response(response))
