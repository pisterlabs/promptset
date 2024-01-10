import openai
import re
import os 

def get_completion(prompt, model="gpt-3.5-turbo", temp=0.8):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(model=model, messages=messages, temperature=temp)
    return response.choices[0].message["content"]

def get_subfolder_names(directory):
    subfolder_names = []
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            subfolder_names.append(os.path.join(root, dir_name).replace(directory + '/', ''))
    return subfolder_names

def remove_java_comments(java_code):
    pattern = r'//.*|/\*[\s\S]*?\*/'
    cleaned_code = re.sub(pattern, '', java_code)
    lines = cleaned_code.splitlines()
    cleaned_lines = [lines[0]]

    for i in range(1, len(lines)):
        if not lines[i].strip() and not lines[i-1].strip():
            continue
        cleaned_lines.append(lines[i])
    
    cleaned_code = '\n'.join(cleaned_lines)
    
    return cleaned_code

def mark_unchanged_lines(input_string):
    output_lines = []
    lines = input_string.split('\n')

    def get_indentation_and_trailing_spaces(line):
        indentation = len(line) - len(line.lstrip())
        trailing_spaces = len(line) - len(line.rstrip())
        return indentation, trailing_spaces

    for line in lines:
        if line.startswith('+') or line.startswith('-'):
            line = line[:1] + '  ' + line[1:]
            output_lines.append(line)
        else:
            indentation, trailing_spaces = get_indentation_and_trailing_spaces(line)
            output_lines.append('o ' + ' ' * indentation + line[indentation:])

    output_string = '\n'.join(output_lines)
    return output_string

def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        return True
    return False

def get_files_in_folder_and_subfolder(folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list
