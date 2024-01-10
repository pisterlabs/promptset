"""
This program is used to find all the functions in a file, to find the code of a specific function, and to replace the code of a specific function.
This enables verbal programming by allowing the user to say the name of the function and what it should do, and then the AI model will find the function in the file and replace the code with the new code based on what the user instructed the AI model to do.
"""

import os
import sys
import openai
import re


def find_functions(filename):
    """
    Finds all the functions in a file.
    :param filename: The name of the file to search in.
    :return: A list of all the functions in the file.
    """
    with open(filename, 'r') as file: # open the file
        code = file.read() # read the file
        functions = re.findall(r'\n\s*def\s+([a-zA-Z0-9_]+)\(', code) # find all the functions in the file and put them in a list.
        for function in functions: # print all the functions in the list
            print(function) # print the function
    return functions



def get_function_code(function_name, file_name):
    """
    Gets the code of a function in a file.
    :param function_name: The name of the function to get the code of.
    :param file_name: The name of the file to search in.
    :return: The code of the function.
    """
    with open(file_name, 'r') as file:
        code = file.read()
    # first find the function definition:
    function_def = re.search(r'\n\s*def\s+' + function_name + r'\(', code, re.MULTILINE)
    if not function_def:
        raise Exception(f"Could not find function definition for {function_name}")
    function_def = function_def.group()
    # now find the beginning and end of the function code:
    function_begin = code.find(function_def)
    function_end = function_begin + len(function_def)
    if next_function_def := re.search(
        r'\n\s*def\s+[a-zA-Z0-9_]+\(', code[function_end:], re.MULTILINE
    ):
        indentation_level = function_def.count('\t') + 1
        # now get the indentation level, we don't want any lines that have a lesser indentation
        while code[function_end:].startswith('\t' * indentation_level):
            function_end += 1
        function_end += next_function_def.start() - 1
    else:
        function_end = len(code)
    # now get the function code:
    return code[function_begin:function_end]


def get_function_code_new(function_name, file_name):
    """
    Gets the code of a function in a file.
    :param function_name: The name of the function to get the code of.
    :param file_name: The name of the file to search in.
    :return: The code of the function.
    """
    with open(file_name, 'r') as file:
        code = file.read()
    # first find the function definition:
    function_def = re.search(r'\n\s*def\s+' + function_name + r'\(', code, re.MULTILINE).group()
    # now find the beginning and end of the function code:
    function_begin, function_end = code.find(function_def), function_begin + len(function_def)
    # to find the end, we need to find the next function definition (or the end of the file):
    next_function_def = re.search(r'\n\s*def\s+[a-zA-Z0-9_]+\(', code[function_end:], re.MULTILINE).group()
    indentation_level = function_def.count('\t') + 1
    # now get the indentation level, we don't want any lines that have a lesser indentation
    while code[function_end:].startswith('\t' * indentation_level):
        function_end += 1
    function_end += next_function_def.start() - 1
    # now get the function code:
    return code[function_begin:function_end]




def replace_function(function_name, file_name, new_code):
    """
    Replaces the code of a function in a file.
    :param function_name: The name of the function to replace.
    :param file_name: The name of the file to search in.
    :param new_code: The new code of the function.
    :return: None
    """
    with open(file_name, 'r') as file:
        code = file.read()
    # first find the function definition:
    function_def = re.search(r'\n\s*def\s+' + function_name + r'\(', code, re.MULTILINE)
    if not function_def:
        raise Exception(f"Could not find function definition for {function_name}")
    function_def = function_def.group()
    # now find the beginning and end of the function code:
    function_begin = code.find(function_def)
    function_end = function_begin + len(function_def)
    if next_function_def := re.search(
        r'\n\s*def\s+[a-zA-Z0-9_]+\(', code[function_end:], re.MULTILINE
    ):
        indentation_level = function_def.count('\t') + 1
        # now get the indentation level, we don't want any lines that have a lesser indentation
        while code[function_end:].startswith('\t' * indentation_level):
            function_end += 1
        function_end += next_function_def.start()
    else:
        function_end = len(code)
    new_code = indent_code(new_code, function_def.count('\t'))
    code = code[:function_begin] + '\n\n' + new_code + code[function_end:] + '\n\n'
    # now write the new code to the file:
    with open(file_name, 'w') as file:
        file.write(code)
    # now ask the user if they would like to make any changes:
    print("Would you like to make any changes to the function? (yes/no)")
    answer = input()

    while answer != 'no':
        print("Would you like to make any changes to the function? (yes/no)")
        answer = input()
    if answer == 'yes':

        replace_function(function_name, file_name, edit_code(get_function_code(function_name, file_name)))




        
        


def indent_code(code, indentation_level):
    """
    Indents a code block by the given indentation level.
    :param code: The code to indent.
    :param indentation_level: The indentation level.
    :return: The indented code.
    """
    return '\t' * indentation_level + code.replace('\n', '\n' + '\t' * indentation_level)




def edit_code(code):
    """
    This function takes in a string of code and returns a string of code that 
    has been edited based on the user's input.
    """
    openai.api_key = "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"
    desired_changes = input("What changes do you want to make to the code? ")
    response = openai.Edit.create(
    model="code-davinci-edit-001",
    input=code,
    instruction=desired_changes,
    temperature=0,
    top_p=1
    )
    new_code = response.choices[0].text
    print(new_code)
    return new_cod








def append_code():
    # first, ask what file I want to append the code to
    file_name = input("What file do you want to append the code to? ").strip()
    line_number = int(input("What line number do you want to append the code to? ").strip())
    instructions = input("What do you want to append? ").strip()
    openai.api_key = "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"
    response = openai.Edit.create(
    model="code-davinci-edit-001",
    input="",
    instruction=instructions,
    temperature=0,
    top_p=1
    )
    new_code = response.choices[0].text
    print(new_code)
    # now append the code to the line number
    # now append the code to the file on the line number that I want
    append_code_to_file(file_name, line_number, new_code)    
    append_more = input("Do you want to append more code? ")
    if append_more.lower() == 'yes':
        append_code()
    else:
        return code
    # now return the code
    return code








def append_code_to_file(file_name, line_number, code):
    """
    Appends the code to the file at the given line number.
    :param file_name: The name of the file.
    :param line_number: The line number to append the code to.
    :param code: The code to append.
    :return: None
    """
    with open(file_name, 'r') as file_:
        file_code = file_.read()
    file_code = file_code.split('\n')
    file_code.insert(line_number, code)
    file_code = '\n'.join(file_code)

    with open(file_name, 'w') as file:
        file.write(file_code)
    return file_code




if __name__ == '__main__':

    function_name = input('Enter a function name: ')
    file_name= input('Enter a file name: ')
    replace_function(function_name, file_name, edit_code(get_function_code(function_name, file_name)))
    # append_code()



















