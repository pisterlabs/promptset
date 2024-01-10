import openai
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def read_templates(folder_path):
    template_data = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.endswith(".py"):
            with open(file_path, 'r') as file:
                template_code = file.read()
                template_data.append({"code": template_code, "description": ""})
    return template_data

def read_template_descriptions(folder_path):
    template_data = read_templates(folder_path)
    for template in template_data:
        description_file_path = os.path.join(folder_path, f"{template['code'][:-3]}_description.txt")
        if os.path.isfile(description_file_path):
            with open(description_file_path, 'r') as file:
                template["description"] = file.read()
    return template_data

def read_additional_details(file_path):
    with open(file_path, 'r') as file:
        additional_details = file.read()
    return additional_details

def write_code_to_file(code, file_path):
    start_marker = "'''"
    end_marker = "'''"

    start_index = code.find(start_marker)
    end_index = code.find(end_marker, start_index + len(start_marker))

    if start_index != -1 and end_index != -1:
        extracted_content = code[start_index + len(start_marker):end_index]
        with open(file_path, "w") as file:
            file.write(extracted_content)
        print(f"Content between triple single quotes saved successfully")
    else:
        with open(file_path, 'w') as file:
            file.write(code)
        print(f"Content saved successfully")

def generate_code_using_templates_and_details(templates_folder, input):
    # Read Python code templates and descriptions from the templates folder
    all_templates = read_template_descriptions(templates_folder)

    # Read additional details from the text file
    additional_details = input

    # Prepare user message with template descriptions
    user_message = f"You are a helpful assistant that generates Python code.\n\n" \
                   f"Only generate Python code using the following templates:\n\n"
    for template in all_templates:
        user_message += f"Template Description: {template['description']}\n"
        user_message += f"Template Code:\n{template['code']}\n\n"

    user_message += f"Additional details:\n{additional_details}"

    # Generate code
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates only Python code. You do not give descriptions or add any comments. You only return Python code."},
            {"role": "user", "content": user_message}
        ]
    )
    generated_code = response.choices[0].message.content

    # Write the generated code to a file
    return generated_code

    print(f"Generated code has been written to {output_file_path}")

