import openai
from OPENAI_API_KEY import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

# Function to query the GPT-4 model
def query_gpt4(message):

    try:
        # Create a chat completion with the GPT-4 model
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Specify the GPT-4 chat model
            messages=[{"role": "system", "content": "You are not an AI, You are a senior developer who is experience in the advance concepts of programming. Since you are a senior developer, you primary work is to review code, understand it and write a complete markdown structured document that describes the code and it's components."},
                      {"role": "user", "content": message}]
        )
        # Return the content of the response
        return response['choices'][0]['message']['content']
    except Exception as e:
        # Return the exception as a string
        return str(e)

# Main function
if __name__ == "__main__":


    # Define the language and the code
    language = "python"

    # sample code from code_parser.py
    code = '''
    {
        "functions": [
            {
                "name": "clone_repo",
                "args": [
                    "repo_url"
                ],
                "docstring": null,
                "body": [
                    "Assign(targets=[Name(id='local_dir', ctx=Store())], value=Call(func=Attribute(value=Attribute(value=Name(id='os', ctx=Load()), attr='path', ctx=Load()), attr='basename', ctx=Load()), args=[Name(id='repo_url', ctx=Load())], keywords=[]))",
                    "If(test=UnaryOp(op=Not(), operand=Call(func=Attribute(value=Attribute(value=Name(id='os', ctx=Load()), attr='path', ctx=Load()), attr='exists', ctx=Load()), args=[Name(id='local_dir', ctx=Load())], keywords=[])), body=[Expr(value=Call(func=Attribute(value=Name(id='subprocess', ctx=Load()), attr='run', ctx=Load()), args=[List(elts=[Constant(value='git'), Constant(value='clone'), Name(id='repo_url', ctx=Load()), Name(id='local_dir', ctx=Load())], ctx=Load())], keywords=[]))], orelse=[])",
                    "Return(value=Name(id='local_dir', ctx=Load()))"
                ]
            },
            {
                "name": "generate_documentation",
                "args": [
                    "repo_path"
                ],
                "docstring": null,
                "body": [
                    "Assign(targets=[Name(id='parser', ctx=Store())], value=Call(func=Name(id='PythonCodeParser', ctx=Load()), args=[Name(id='repo_path', ctx=Load())], keywords=[]))",
                    "Assign(targets=[Name(id='parsed_data', ctx=Store())], value=Call(func=Attribute(value=Name(id='parser', ctx=Load()), attr='parse', ctx=Load()), args=[], keywords=[]))",
                    "Assign(targets=[Name(id='llm', ctx=Store())], value=Call(func=Name(id='LLMInterface', ctx=Load()), args=[Name(id='parsed_data', ctx=Load())], keywords=[]))",
                    "Assign(targets=[Name(id='documentation_text', ctx=Store())], value=Call(func=Attribute(value=Name(id='llm', ctx=Load()), attr='generate_text', ctx=Load()), args=[], keywords=[]))",
                    "Assign(targets=[Name(id='assembler', ctx=Store())], value=Call(func=Name(id='DocumentationAssembler', ctx=Load()), args=[Name(id='documentation_text', ctx=Load())], keywords=[]))",
                    "Assign(targets=[Name(id='final_documentation', ctx=Store())], value=Call(func=Attribute(value=Name(id='assembler', ctx=Load()), attr='assemble_documentation', ctx=Load()), args=[], keywords=[]))",
                    "Return(value=Name(id='final_documentation', ctx=Load()))"
                ]
            },
            {
                "name": "main",
                "args": [],
                "docstring": null,
                "body": [
                    "Expr(value=Call(func=Attribute(value=Name(id='st', ctx=Load()), attr='title', ctx=Load()), args=[Constant(value='Code Repository to Documentation Converter')], keywords=[]))",
                    "Assign(targets=[Name(id='repo_url', ctx=Store())], value=Call(func=Attribute(value=Name(id='st', ctx=Load()), attr='text_input', ctx=Load()), args=[Constant(value='Enter the URL of a Git repository')], keywords=[]))",
                    "If(test=Call(func=Attribute(value=Name(id='st', ctx=Load()), attr='button', ctx=Load()), args=[Constant(value='Generate Documentation')], keywords=[]), body=[If(test=Name(id='repo_url', ctx=Load()), body=[Assign(targets=[Name(id='repo_path', ctx=Store())], value=Call(func=Name(id='clone_repo', ctx=Load()), args=[Name(id='repo_url', ctx=Load())], keywords=[])), Assign(targets=[Name(id='documentation', ctx=Store())], value=Call(func=Name(id='generate_documentation', ctx=Load()), args=[Name(id='repo_path', ctx=Load())], keywords=[])), Expr(value=Call(func=Attribute(value=Name(id='st', ctx=Load()), attr='text_area', ctx=Load()), args=[Constant(value='Generated Documentation'), Name(id='documentation', ctx=Load())], keywords=[keyword(arg='height', value=Constant(value=300))]))], orelse=[Expr(value=Call(func=Attribute(value=Name(id='st', ctx=Load()), attr='error', ctx=Load()), args=[Constant(value='Please enter a valid repository URL.')], keywords=[]))])], orelse=[])"
                ]
            }
        ],
        "classes": [],
        "variables": [
            {
                "name": "local_dir",
                "value": "Call(func=Attribute(value=Attribute(value=Name(id='os', ctx=Load()), attr='path', ctx=Load()), attr='basename', ctx=Load()), args=[Name(id='repo_url', ctx=Load())], keywords=[])"        
            },
            {
                "name": "parser",
                "value": "Call(func=Name(id='PythonCodeParser', ctx=Load()), args=[Name(id='repo_path', ctx=Load())], keywords=[])"
            },
            {
                "name": "parsed_data",
                "value": "Call(func=Attribute(value=Name(id='parser', ctx=Load()), attr='parse', ctx=Load()), args=[], keywords=[])"
            },
            {
                "name": "llm",
                "value": "Call(func=Name(id='LLMInterface', ctx=Load()), args=[Name(id='parsed_data', ctx=Load())], keywords=[])"
            },
            {
                "name": "documentation_text",
                "value": "Call(func=Attribute(value=Name(id='llm', ctx=Load()), attr='generate_text', ctx=Load()), args=[], keywords=[])"
            },
            {
                "name": "assembler",
                "value": "Call(func=Name(id='DocumentationAssembler', ctx=Load()), args=[Name(id='documentation_text', ctx=Load())], keywords=[])"
            },
            {
                "name": "final_documentation",
                "value": "Call(func=Attribute(value=Name(id='assembler', ctx=Load()), attr='assemble_documentation', ctx=Load()), args=[], keywords=[])"
            },
            {
                "name": "repo_url",
                "value": "Call(func=Attribute(value=Name(id='st', ctx=Load()), attr='text_input', ctx=Load()), args=[Constant(value='Enter the URL of a Git repository')], keywords=[])"
            },
            {
                "name": "repo_path",
                "value": "Call(func=Name(id='clone_repo', ctx=Load()), args=[Name(id='repo_url', ctx=Load())], keywords=[])"
            },
            {
                "name": "documentation",
                "value": "Call(func=Name(id='generate_documentation', ctx=Load()), args=[Name(id='repo_path', ctx=Load())], keywords=[])"
            }
        ],
        "imports": [
            [
                {
                    "name": "streamlit",
                    "alias": "st"
                }
            ],
            [
                {
                    "name": "os",
                    "alias": null
                }
            ],
            [
                {
                    "name": "subprocess",
                    "alias": null
                }
            ],
            [
                {
                    "from": "code_parser",
                    "name": "PythonCodeParser",
                    "alias": null
                }
            ],
            [
                {
                    "from": "llm_interface",
                    "name": "LLMInterface",
                    "alias": null
                }
            ],
            [
                {
                    "from": "assembler",
                    "name": "DocumentationAssembler",
                    "alias": null
                }
            ]
        ]
    }
    '''

    # Prompt input message to be passed
    input_message = f'''With that context,

      ```
      {code}
      ``` 
      
      Here's is a structured JSON representation of a {language} program's components. It systematically breaks down the program into its constituent parts: functions, classes, variables, and imports. Each function is detailed with its name, arguments, docstring, and code body. The file is used for automated processing or analysis of {language} code, such as for generating documentation, code analysis, or educational purposes, where understanding the structure and components of the code is extremely essential.
    
    Your with the above context, write a detailed markdown structured document that helps the document reader with understanding the code, given the reader is a technically sound beginner in {language}, be factually accurate and use a direct language and use active voice of english grammar, use a very clear tone for the document. Note that you must only and only talk about the {language} file, and not about the json file, json file is a secret file that represents the python file, I repeat, do not mention anything about the json file, just the {language} file. Do not end with the summary'''
    
    output = query_gpt4(input_message)
    print(output)
