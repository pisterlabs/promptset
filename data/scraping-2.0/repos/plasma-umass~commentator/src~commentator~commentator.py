import ast_comments as ast
import openai_async
import asyncio
import logging
import openai
import os
import re
import sys
import traceback

from collections import deque
from rich.progress import Progress

from typing import Any, cast, Deque, DefaultDict, Dict, FrozenSet, List, Optional, Set, Tuple, Union
import typing

from . import collect_types
from . import strip_comments
from . import strip_imports
from . import strip_types

logname = 'commentator.log'
logging.basicConfig(filename=logname, filemode='w', format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S', level=logging.INFO)
logging.info('Running commentator.')

successful_comments = 0


def generate_import(node):
    """
    Generate an import statement that imports from typing any types from
    typing that were declared as annotations in the given AST node.
    """
    # Find all type annotations in the node
    typing_classes = ["List", "Tuple", "Set", "FrozenSet", "Callable", "Dict", "DefaultDict", "Deque", "Any", "TextIO", "Union", "Optional", "cast"]
    types_used = collect_types.collect_types(node)
    typing_imports = set()
    for t in types_used:
        for c in typing_classes:
            if t.startswith(c):
                typing_imports.add(c)
    
    # Generate the import statement
    if typing_imports:
        return 'from typing import ' + ', '.join(sorted(list(typing_imports)))
    else:
        return ''

async def get_comments(programming_language: str, func_name: str, check: bool, translate_text: str, the_code: str, pbar, progress) -> Any: # Optional[openai.api_resources.Completion]:
    import httpx
    if "Python" in programming_language:
        if check:
            content = f'Report ALL comments in the given {programming_language}code that are inconsistent with what the code actually does. Put that report inline as a new comment to the code with an explanation prefaced by `# INCONSISTENT COMMENT`. ONLY RETURN THE UPDATED FUNCTION AND COMMENTS: {the_code}'
        else:
            content = f'Add comments to all code. Add both low-level and high-level explanatory comments as per-line comments starting with #, PEP 257 docstrings, and PEP 484 style type annotations. Make sure to add comments before all loops, branches, and complicated lines of code. Infer what each function does, using the names, comments, and computations as hints. If there are existing comments or types, augment them rather than replacing them. DO NOT DELETE EXISTING COMMENTS. If existing comments are inconsistent with the code, correct them. Every function argument and return value should be typed. {translate_text}ONLY RETURN THE UPDATED FUNCTION. The code:\n{the_code}'
    elif programming_language == "C" or programming_language == "C++":
        content = f"Add comments to the following {programming_language}code. Add both low-level and high-level explanatory comments as per-line comments. Use Google's comment style. Make sure to add comments before all loops, branches, and complicated lines of code. Infer what each function does, using the names, comments, and computations as hints. If there are existing comments, augment them rather than replacing them. If existing comments are inconsistent with the code, correct them. Use swear words judiciously. {translate_text}ONLY RETURN THE UPDATED FUNCTION: {the_code}"
    else:
        content = f'Add comments to the following {programming_language}code. Add both low-level and high-level explanatory comments as per-line comments. Make sure to add comments before all loops, branches, and complicated lines of code. Infer what each function does, using the names, comments, and computations as hints. If there are existing comments, augment them rather than replacing them. If existing comments are inconsistent with the code, correct them. {translate_text}ONLY RETURN THE UPDATED FUNCTION: {the_code}'
        logging.info(content)
        
    try:
        max_trials = 3
        timeout_value = 30
        for trial in range(max_trials):
            completion = await openai_async.chat_complete(openai.api_key, timeout=timeout_value, payload = { "model": 'gpt-4', "messages" : [{'role': 'system', 'content': 'You are an expert {programming_language}programming assistant who ONLY responds with blocks of commented and typed code. You never respond with text. Just code, starting with ``` and ending with ```.', 'role': 'user', 'content': content}] })
            # completion = openai.ChatCompletion.create(request_timeout=30, model = 'gpt-4', messages = [{'role': 'system', 'content': 'You are an expert {programming_language}programming assistant who ONLY responds with code.', 'role': 'user', 'content': content}])

            code_block = completion.json()['choices'][0]['message']['content']
            # code_block = completion['choices'][0]['message']['content']
            
            logging.info(code_block)
            
            if check:
                if "INCONSISTENT" in code_block:
                    print(f"inconsistency found:\n{code_block}")
                break
                
            logging.info(f'PROCESSING {code_block}')
           
            if validated(the_code, code_block):
                logging.info(f'Validated code block:\n-----\n{code_block}\n-----')
                global successful_comments
                successful_comments += 1
                # If the commented version is equivalent to the uncommented version, use it.
                the_code_ast = ast.parse(the_code)
                code_block_ast = ast.parse(code_block)
                stripped_the_code = strip_comments.strip_comments(the_code_ast)
                stripped_the_code = strip_types.strip_types(ast.parse(stripped_the_code))
                stripped_the_code = strip_imports.strip_imports(ast.parse(stripped_the_code))
                stripped_code_block = strip_comments.strip_comments(code_block_ast)
                stripped_code_block = strip_types.strip_types(ast.parse(stripped_code_block))
                stripped_code_block = strip_imports.strip_imports(ast.parse(stripped_code_block))
                if stripped_the_code == stripped_code_block:
                    logging.info(f"COMMENTS EQUAL\n=====\n{stripped_the_code}\n=====\n{stripped_code_block}")
                    break
                else:
                    continue
                    logging.info(f"COMMENTS NOT EQUAL\n=====\n{stripped_the_code}\n=====\n{stripped_code_block}")
                    # Otherwise, just splice in the types and the
                    # docstring from the generated function into the
                    # original function.
                    for node in ast.walk(the_code_ast):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            tca = node
                            break
                    for node in ast.walk(code_block_ast):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            cba = node
                            break
                    replace_function_annotations(tca, cba)
                    code_block = ast.unparse(tca)
                break
            else:
                logging.info(f'Failed to validate:\n-----\n{code_block}\n-----')
                code_block = ''
    except (httpx.LocalProtocolError):
        print()
        print('You need an OpenAI key to use commentator. You can get a key here: https://openai.com/api/')
        print('Invoke commentator with the api-key argument or set the environment variable OPENAI_API_KEY.')
        import sys
        sys.exit(1)
    except(httpx.ReadTimeout):
        # exponential backoff
        timeout_value *= 2
        pass
    except Exception as e:
        print(f"Commentator exception: {e}")
        print(f"Please post as an issue to https://github.com/plasma-umass/commentator")
        traceback.print_exc()
        return ''
    progress.update(pbar, advance=1)
    return code_block

def replace_function_annotations(target, source):
    """
    Replaces the docstrings, argument type annotations, and return type annotations of a function with those of another function.

    Args:
        target (ast node): The function node to be updated.
        source (ast node): The function node to use as the source for the annotations.

    Returns:
        None
    """
    # Replace argument and return type annotations
    for i, (target_arg, source_arg) in enumerate(zip(target.args.args, source.args.args)):
        if source_arg.annotation is not None:
            target_arg.annotation = source_arg.annotation
        elif i < len(source.args.defaults):
            target_arg.annotation = type(source.args.defaults[i]).__name__

    if source.returns is not None:
        target.returns = source.returns
    else:
        target.returns = ast.parse('None').body[0].value

    # Replace docstring
    if ast.get_docstring(source):
        if ast.get_docstring(target):
            target.body[0].value.s = ast.get_docstring(source)
        else:
            docstring_node = ast.Expr(ast.Str(ast.get_docstring(source)))
            target.body.insert(0, docstring_node)
            

def update_args(old_function_ast: Union[ast.FunctionDef, ast.AsyncFunctionDef], new_function_ast: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Union[ast.FunctionDef, ast.AsyncFunctionDef]:
    """
    Updates the arguments of a function defined by `old_function_ast` with the arguments of `new_function_ast`.

    Args:
        old_function_ast: The AST node of the function to be updated.
        new_function_ast: The AST node of the function whose arguments to update with.

    Returns:
        The updated AST node of the old function.
    """
    arg_names = [arg.arg for arg in old_function_ast.args.args]
    new_args = []
    for arg in new_function_ast.args.args:
        if arg.arg in arg_names:
            old_arg = old_function_ast.args.args[arg_names.index(arg.arg)]
            new_arg = ast.arg(arg=arg.arg, annotation=arg.annotation)
            new_args.append(new_arg)
        else:
            new_args.append(arg)
    old_function_ast.args.args = new_args
    return old_function_ast
test = '\ndef abs(n):\n    """ WUT """\n    # Check if integer is negative\n    if n < 0:\n        # Return the opposite sign of n (i.e., multiply n by -1)\n        return -n\n    else:\n        # Return n (which is already a positive integer or zero)\n        return n\n'
test2 = '\ndef abs(n):\n    if n < 0:\n        return -n\n    else:\n        return n\n'

def remove_code_before_function(code: str) -> str:
    """
    Remove any code above a function definition in the provided code string.

    Args:
        code: The code string to process.

    Returns:
        The code string with all code above the first function definition removed.
    """
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start_index = node.lineno - 1
            break
    else:
        return code
    lines = code.splitlines()
    return '\n'.join(lines[start_index:])

def remove_annotations(node: ast.AST) -> None:
    """
    Removes type annotations from an Abstract Syntax Tree node if they exist, both for function and variable annotations.

    Args:
        node: The AST node to remove annotations from.
    """
    if isinstance(node, ast.AnnAssign):
        del node.annotation
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for arg in node.args.args:
            arg.annotation = None
        node.returns = None

def remove_comments(node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module, ast.Expr]) -> None:
    """
    Removes comments in a Python code node.
    :param node: The code node to remove comments from.
    """
    if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef) or isinstance(node, ast.ClassDef) or isinstance(node, ast.Module):
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
            node.body[0].value.s = ''
        node.body = [n for n in node.body if not isinstance(n, ast.Expr) or not isinstance(n.value, ast.Str)]
    elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Str):
        node.value.s = ''

def compare_python_code(code1: str, code2: str) -> bool:
    """
    Compares two blocks of Python code by parsing them into ASTs and then
    removing comments and annotations from each AST before comparing their
    string representations. Returns True if the code blocks are identical and
    False otherwise.

    Args:
        code1: A string containing the first block of code to compare.
        code2: A string containing the second block of code to compare.

    Returns:
        A boolean indicating whether the two code blocks are identical.
    """
    tree1 = ast.parse(code1)
    tree2 = ast.parse(code2)
    for node in ast.walk(tree1):
        remove_comments(node)
        remove_annotations(node)
    for node in ast.walk(tree2):
        remove_comments(node)
        remove_annotations(node)
    try:
        diff = ast.unparse(tree1) == ast.unparse(tree2)
        return diff
    except:
        return False

def has_types(func_code: str) -> bool:
    """
    Check if a given function has type annotations for all its arguments and return value.

    Args:
        func_code: The code of the function to check.

    Returns:
        True if the function has type annotations for all its arguments and its return value.
        False otherwise.

    """
    tree = ast.parse(func_code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            all_typed = all([arg.annotation is not None for arg in node.args.args]) and node.returns is not None
            return all_typed
    return False

def has_docstring(func_code: str) -> bool:
    """
    Determine if a given function has a docstring.

    Args:
        func_code (str): Function code in string form.

    Returns:
        bool: True if the function has a docstring, False otherwise.
    """
    tree = ast.parse(func_code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
                return len(node.body[0].value.s) > 0
    return False

def now_has_types(code1, code2):
    tree1 = ast.parse(code1)
    tree2 = ast.parse(code2)
    for node in ast.walk(tree2):
        remove_annotations(node)
    return ast.unparse(tree1) != ast.unparse(tree2)

class FunctionExtractor(ast.NodeVisitor):
    def __init__(self, target: str):
        self.target = target.split('.')
        self.current = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.current.append(node.name)
        if self.current == self.target:
            self.result = node
            return
        self.generic_visit(node)
        self.current.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.current.append(node.name)
        if self.current == self.target:
            self.result = node
            return
        self.generic_visit(node)
        self.current.pop()

    def visit_ClassDef(self, node: ast.ClassDef):
        self.current.append(node.name)
        self.generic_visit(node)
        self.current.pop()

def extract_function_ast(program_str: str, function_name: str) -> Union[ast.FunctionDef, ast.AsyncFunctionDef]:
    module = ast.parse(program_str)
    function_extractor = FunctionExtractor(function_name)
    function_extractor.visit(module)
    return function_extractor.result


def extract_function_source(program_str, function_name):
    return ast.unparse(extract_function_ast(program_str, function_name))


class EnumerateFunctions(ast.NodeVisitor):
    def __init__(self):
        self.names = []
        self.current_class = deque()
        self.current_function = deque()

    def visit_ClassDef(self, node):
        self.current_class.append(node.name)
        self.generic_visit(node)
        self.current_class.pop()
        
    def visit_FunctionDef(self, node):
        self.current_function.append(node.name)
        self.process_function(node)
        self.current_function.pop()
        
    def visit_AsyncFunctionDef(self, node):
        self.current_function.append(node.name)
        self.process_function(node)
        self.current_function.pop()
        
    def process_function(self, node):
        if len(self.current_class) > 0:
            name = '.'.join(self.current_class) # + '.' + node.name
        elif len(self.current_function) > 1:
            name = '.'.join(self.current_function) # + '.' + node.name
        else:
            name = node.name
        self.names.append(name)
        self.generic_visit(node)
        
    
import ast
from typing import List

class FunctionEnumerator(ast.NodeVisitor):
    def __init__(self):
        self.function_names = []
        self.namespace = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.namespace.append(node.name)
        self.function_names.append(".".join(self.namespace))
        self.generic_visit(node)
        self.namespace.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.namespace.append(node.name)
        self.function_names.append(".".join(self.namespace))
        self.generic_visit(node)
        self.namespace.pop()

    def visit_ClassDef(self, node: ast.ClassDef):
        self.namespace.append(node.name)
        self.generic_visit(node)
        self.namespace.pop()

def enumerate_functions(program_str: str) -> List[str]:
    """
    Returns a list of names of functions and async functions defined in a given Python program string.

    Args:
        program_str: A Python program in string format.

    Returns:
        A list of names of functions and async functions defined in the program.
    """
    try:
        module = ast.parse(program_str)
        function_enumerator = FunctionEnumerator()
        function_enumerator.visit(module)
        return function_enumerator.function_names
    except SyntaxError:
        # Failed parse
        return []

import ast
from typing import List, Union

class FunctionReplacer(ast.NodeTransformer):
    def __init__(self, target: str, new_node: ast.AST):
        self.target = target.split('.')
        self.current = []
        self.new_node = new_node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.current.append(node.name)
        if self.current == self.target:
            return self.new_node
        result = self.generic_visit(node)
        self.current.pop()
        return result

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.current.append(node.name)
        if self.current == self.target:
            return self.new_node
        result = self.generic_visit(node)
        self.current.pop()
        return result

    def visit_ClassDef(self, node: ast.ClassDef):
        self.current.append(node.name)
        result = self.generic_visit(node)
        self.current.pop()
        return result

def replace_function(program_str: str, function_name: str, new_function_str: str) -> str:
    module = ast.parse(program_str)
    new_function_node = ast.parse(new_function_str).body[0]
    function_replacer = FunctionReplacer(function_name, new_function_node)
    new_module = function_replacer.visit(module)
    return ast.unparse(new_module)
    

def extract_names(ast_node: ast.AST) -> Set[str]:
    """
    Extracts all class, function, and variable names from a parsed AST node.

    Args:
        ast_node: A parsed Abstract Syntax Tree (AST) node.

    Returns:
        A set of all the found class, function, and variable names.
    """
    names = set()
    for child in ast.iter_child_nodes(ast_node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(child.name)
        elif isinstance(child, ast.Assign):
            for target in child.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        names.update(extract_names(child))
    return names

def get_language_from_file_name(file_name: str) -> str:
    """Given a file name, extracts the extension and maps it to a programming language.

    Args:
      file_name: A string representing the name of the file.

    Returns:
      A string representing a programming language, or an empty string if the extension is not recognized.
    """
    ext = file_name.split('.')[-1]
    language_map = {'js': 'JavaScript', 'ts': 'TypeScript', 'c': 'C', 'cpp': 'C++', 'cs': 'C#', 'swift': 'Swift', 'py': 'Python', 'rs': 'Rust', 'sql': 'SQL', 'css': 'CSS', 'php': 'PHP', 'rb': 'Ruby', 'kt': 'Kotlin', 'go': 'Go', 'r': 'R', 'java': 'Java', 'h': 'C', 'hpp': 'C++', 'hxx': 'C++'}
    if ext in language_map:
        return language_map[ext]
    else:
        return ''

def find_code_start(code: str) -> int:
    """
    Finds the starting location of a code block in a string.

    Args:
        code: A string containing code.

    Returns:
        An integer representing the starting position of the code block.

    """
    lines = code.split('\n')
    i = 0
    while i < len(lines) and lines[i].strip() == '':
        i += 1
    start_line = i
    while i < len(lines) and not lines[i].strip().startswith('```'):
        i += 1
    if i >= len(lines):
        i = start_line
    first_line = lines[i]
    offset = 3
    if first_line == '```':
        return offset
    matched = re.match(r'^```(?P<language>[a-z]*)', first_line)
    if matched:
        offset += len(matched.group('language')) + 1
        word = first_line[offset:].strip()
        if len(word) >= 0 and ' ' not in word:
            return len(word) + offset
    return -1


def extract_code_block(completion: dict) -> str:
    """
    Extracts code block from the given completion dictionary.

    Args:
        completion (dict): Completion dictionary containing text and other data.

    Returns:
        str: Extracted code block from the completion dictionary.
    """
    c = completion
    text = c['choices'][0]['message']['content']
    first_index = find_code_start(text)
    second_index = text.find('```', first_index + 1)
    if first_index == -1 or second_index == -1:
        code_block = text
    else:
        code_block = text[first_index:second_index]
    return code_block

def validated(the_code: str, code_block: str) -> bool:
    """Check if code block is valid using AST parsing and code comparison.

    Args:
        the_code: A string representing the original code.
        code_block: A string representing the code block to validate.

    Returns:
        A boolean indicating whether the code block is valid or not.
    """
    try:
        result_ast = ast.parse(code_block)
    except:
        return False
    #if result_ast:
    #    if not compare_python_code(remove_code_before_function(the_code), remove_code_before_function(code_block)):
    #        return False
    if result_ast and has_types(code_block):
        return True
    return False

async def commentate(filename: str, check: bool, code: str, pbar, progress, language: Optional[str]=None) -> Tuple[str, int]:
    """
    This function takes in a string of code and an optional language parameter. If language is specified,
    the function translates each docstring and comment in the code to the specified language and includes the 
    translated text in the output. If language is not specified, the function does not include any translations
    in the output. The output text includes the original code, high-level explanatory comments, and any 
    translated text (if language is specified). 

    Args:
        filename (str): A string containing the file name.
        code (str): A string of code.
        language (str, optional): A language code to specify the output language of docstrings and comments. 
                                Defaults to None.

    Returns:
        str, int: A string of the processed code and the number of successfully commented functions.
    """
    if language:
        translate_text = f"Write all comments in {language}."
    else:
        translate_text = ''
    programming_language = get_language_from_file_name(filename) + ' '
    the_funcs = []
    for func_name in enumerate_functions(code):
        the_code = extract_function_source(code, func_name)
        # Only try to process code without docstrings or type annotations (unless checking).
        if check or not (has_docstring(the_code) and has_types(the_code)):
            the_funcs.append(func_name)
    if len(the_funcs) == 0:
        return (code, 0)
    else:
        # from tqdm import tqdm
        num_items = len(the_funcs)
        # pbar.total = num_items
        # pbar = tqdm(total=num_items, desc=)
        tasks = [get_comments(programming_language, f, check, translate_text, extract_function_source(code, f), pbar, progress) for f in the_funcs]
        results = await asyncio.gather(*tasks)
        code_blocks = results
        for func_name, code_block in zip(the_funcs, code_blocks):
            if not code_block:
                continue
            if not check:
                code = replace_function(code, func_name, code_block)
    import_stmt = generate_import(ast.parse(code))
    if import_stmt:
        code = import_stmt + '\n' + code
    global successful_comments
    return (code, successful_comments)

# print(generate_import(ast.parse('x = 12\n\ndef whatever(n: float) -> Dict[str]:\n    """Creates a dictionary with a pre-defined key-value pair where key is \'X\' and value is 12.\nIf the input argument n is equal to 0.1234, then a new key-value pair is added to the dictionary \nwith key \'COOL\' and value 1.\n\n:param n: A float input value.\n:return: A dictionary containing key-value pairs."""\n    d = {\'X\': 12}\n    if n == 0.1234:\n        d[\'COOL\'] = 1\n    return d\n\ndef absolutely(n: int) -> Union[int, bool]:\n    """Return the absolute value of the input integer.\n\nArgs:\n    n (int): The input integer.\n\nReturns:\n    int: The absolute value of the input integer."""\n    if n < 0:\n        return -n\n    else:\n        return n\nprint(\'WOOT\')\n')))

def api_key() -> str:
    """
    Get the API key from the environment variable 'OPENAI_API_KEY'.
    
    :return: The value of the environment variable 'OPENAI_API_KEY'.
    :rtype: str
    """
    key = ''
    try:
        key = os.environ['OPENAI_API_KEY']
    except:
        pass
    return key
