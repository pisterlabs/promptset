import sys
import ast
import openai

# Read input from command-line argument
# input_argument = sys.argv[1]
# code_snippet = input_argument


import ast
import openai
import os

# Your OpenAI GPT-3 API key
api_key = os.environ.get('API_KEY')

# Define a Python code snippet that you want to document
code_snippet = """
import torch

def get_entropy_of_dataset(tensor:torch.Tensor):
    num_samples = tensor.shape[0]
    _, classes_counts = torch.unique(tensor[:,-1],return_counts=True)
    if classes_counts[1]==0 or classes_counts[1]==1:
        return 0
    ent = -1*(classes_counts[0]/num_samples)*torch.log2(classes_counts[0]/num_samples)
    ent -= (classes_counts[1]/num_samples)*torch.log2(classes_counts[1]/num_samples)
    return ent.item()

def get_avg_info_of_attribute(tensor:torch.Tensor, attribute:int):
    num_samples = tensor.shape[0]
    classes, classes_counts = torch.unique(tensor[:,attribute],return_counts=True)
    yeses = torch.zeros(classes.shape[0])
    nos = torch.zeros(classes.shape[0])
    for i in range(tensor.shape[0]):
        att_value = tensor[i,attribute]
        # index = classes.index(att_value)
        index = torch.where(classes==att_value)[0]
        if tensor[i,-1]==1:
            yeses[index] += 1
        else:
            nos[index] += 1

    ent = 0
    for j in range(yeses.shape[0]):
        total = yeses[j] + nos[j]
        if(yeses[j]==total or yeses[j]==0):
            continue
        local_ent = 0
        local_ent -= 1*(yeses[j]/total)*torch.log2(yeses[j]/total)
        local_ent -= 1*(nos[j]/total)*torch.log2(nos[j]/total)  
        ent += (total/num_samples)*local_ent
    # print("Entropy is ",ent)
    return ent

def get_information_gain(tensor:torch.Tensor, attribute:int):
    main_entropy = get_entropy_of_dataset(tensor)
    attribute_entropy = get_avg_info_of_attribute(tensor,attribute)
    return (main_entropy - attribute_entropy)
"""

# Parse the code snippet into an AST
parsed_code = ast.parse(code_snippet)

# Extract the relevant code information (e.g., function name, docstring)
code_info = ""
for node in ast.walk(parsed_code):
    if isinstance(node, ast.FunctionDef):
        code_info += f"Function: {node.name}\n"
        if node.body:
            for statement in node.body:
                if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Str):
                    code_info += f"Description: {statement.value.s}\n"
                    print(code_info)


# Prompt for overall implementation
prompt = f"Generate a function and method documentation of the following code:\n\n{code_info}\n\nDocumentation:"

openai.api_key = api_key

# Generate overall documentation using GPT-3
response = openai.Completion.create(
    engine="text-davinci-002",  # Choose the appropriate GPT-3 model
    prompt=prompt,
    max_tokens=100,  # Adjust the length as needed
)
generated_documentation = response.choices[0].text
print("Overall documentation",generated_documentation)



def extract_functions_from_code(code):
    # Parse the input code into an abstract syntax tree (AST).
    parsed_code = ast.parse(code)

    # Create a list to store extracted function nodes.
    function_nodes = []
    def process_function(node):
        if isinstance(node, ast.FunctionDef):
            function_nodes.append(node)
    for node in ast.walk(parsed_code):
        process_function(node)

    return function_nodes


extracted_functions = extract_functions_from_code(code_snippet)

prompt_full = f"""Your task is to document code for a company. Give a function and method documentation for the following functions: {generated_documentation} 
Produce a documentation explaining the use of the function given below and its implementation in high verbosity: {extracted_functions[0]}
Make it understandable to a developer """

response_func = openai.Completion.create(
    engine="text-davinci-002",  # Choose the appropriate GPT-3 model
    prompt=prompt_full,
    max_tokens=100,  # Adjust the length as needed
)
func_documentation = response_func.choices[0].text
print(f"Function {extracted_functions[0]}",func_documentation)
    



# prompt2 = f"""Document the {method_name} method of the {class_name} class.
# Method Description:
# Parameters:
# Return Value:
# Example Usage:
# """

user_content = """
- Overview of the product
- Installation and setup instructions
- How to use the product
- Frequently asked questions (FAQs)
- Troubleshooting tips
"""
user_section_prompt = "### User Section\n\nPlease find below the information for end-users:\n"

prompt_user = f"""Your task is to document code for a company. Give a function and method documentation for the following functions: {generated_documentation} 
Produce a documentation explaining the use of the function given below and its implementation in high verbosity: {extracted_functions[1]}
Make it understandable to a user in the format: {user_content} """

response_user = openai.Completion.create(
    engine="text-davinci-002",  # Choose the appropriate GPT-3 model
    prompt=prompt_user,
    max_tokens=1500,  # Adjust the length as needed
)


developer_content = """
- Technical architecture
- APIs and SDKs
- Integration guides
- Code samples
- Contributing guidelines
"""
prompt_dev = f"""Your task is to document software for a company. Give a function and method documentation for the following functions: {generated_documentation} 
Produce a documentation explaining the use of the function given below and its implementation in high verbosity: {extracted_functions[0]}
Make it understandable to a developer who wants to work on the software in the format: {developer_content} """

response_dev = openai.Completion.create(
    engine="text-davinci-002",  # Choose the appropriate GPT-3 model
    prompt=prompt_dev,
    max_tokens=1500,  # Adjust the length as needed
)