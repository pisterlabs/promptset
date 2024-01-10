# Copyright CEA France
# Author : Yoann CURE
# PHELIQS / NPSC
# This program provides a variety of functions for parsing, formatting, and generating code using Python and OpenAI's
# GPT-3. It includes functions for retrieving an OpenAI API key, adding docstrings to Python code, parsing and
# extracting functions from source code, generating detailed docstrings using GPT-3, checking and compiling Python
# code, and more. Overall, this program could be useful for programmers looking to streamline their code formatting and
# documentation processes.
import os
import shutil
import subprocess
import time
import openai
import re
import ast
from openai.error import OpenAIError
import autopep8
from distutils import dir_util


API_langage = {"en": "You are able to write doc-strings respecting PEP 7 and google style convention by adding them to the "
               "python function provided as input. the response should not be in a comment block, "
               "should only contain the function and the docstring without any other comments. "
               "You then need to delete the first and last line of the response.",
               "fr": "Vous êtes capable d’écrire des doc-string en respectant la PEP 7 et google style convention en les "
               "ajoutant à la fonction python fournie en entrée. la réponse ne doit pas être dans un "
               "bloc de commentaire, ne doit contenir que la fonction et le docstring sans aucun "
               "autre commentaire. Vous devez ensuite supprimer la première et dernière ligne de la réponse."}


# API key of openai
def get_openai_api_key():
    """
    This function retrieves the OpenAI API key from a text file named "openAI_key.txt". It then checks if the file exists, reads the API key from the file, sets it as the current OpenAI API key, and finally validates the key by attempting to list the available models using the OpenAI API.

    If the file does not exist, it returns None.

    If the API key is successfully validated, it returns the API key, otherwise it also returns None and prints an error message indicating that the API key is not valid, along with the error message describing the issue.

    Args:

    Returns:
        api_key (str): The OpenAI API key, may be None if the key is invalid or the file does not exist.
    """
    api_key_path = "openAI_key.txt"
    # Vérifier si le fichier existe
    if not os.path.isfile(api_key_path):
        print(f"File {api_key_path} doesn't exist.")
        return None

    # Lire la clé d'API depuis le fichier
    with open(api_key_path, "r") as f:
        api_key = f.read().strip()

    # Vérifier si la clé d'API est valide
    openai.api_key = api_key
    try:
        models = openai.Model.list()
        print("API key OpenAI is valid.")
        return api_key
    except Exception as e:
        print(f"Not a valid API key : {e}")
        return None


class ReplaceEmptyValue(ast.NodeTransformer):
    def visit(self, node):
        """
        The function `visit` takes one parameter:

        - `node`: an instance of the `ast` module representing a node in an Abstract Syntax Tree.

        The function checks if the `node` is an instance of `ast.Expr` and if the `value` attribute for the `node` is `None`. If this condition is true, it returns a new `ast.Expr` instance with a `ast.Constant` instance with a `value` attribute of `None`. If the condition is false, it returns the original `node` instance.

        The `ast` argument is a built-in module for Python that is used to generate an Abstract Syntax Tree (AST) from Python code. The AST is a tree-like representation of the structure of the code that can be analyzed or manipulated programmatically.
        """
        if isinstance(node, ast.Expr) and node.value is None:
            return ast.Expr(value=ast.Constant(value=None))
        return node


def generate_uml_diagram(code_str, output_file):
    """
    Cette méthode prend en entrée une chaîne de code Python et un nom de fichier de sortie,
    génère un diagramme de classe UML à partir du code, et le sauvegarde au format DOT dans un fichier.
    """
    # Création d'un fichier temporaire pour stocker le code Python
    tmp_file = 'tmp.py'
    with open(tmp_file, 'w') as f:
        f.write(code_str)

    # Appel à Pyreverse pour générer le diagramme UML
    cmd = f'pyreverse -o dot {tmp_file}'
    subprocess.run(cmd.split(), check=True)

    # Renommage du fichier de sortie généré par Pyreverse
    dot_file = f'{tmp_file}_dot.png'
    subprocess.run(f'mv classes.dot {dot_file}'.split(), check=True)

    # Lecture du contenu du fichier DOT
    with open(dot_file, 'r') as f:
        dot_content = f.read()

    # Sauvegarde du fichier DOT dans un fichier de sortie
    with open(output_file, 'w') as f:
        f.write(dot_content)

    # Suppression des fichiers temporaires
    subprocess.run(f'rm {tmp_file} {dot_file}'.split(), check=True)


def generate_prompt(dot_file_path, code_string):
    # Ouvrir le fichier DOT et le lire en tant que chaîne de caractères
    with open(dot_file_path, "r") as dot_file:
        dot_string = dot_file.read()

    # Extraire les noms des classes et des méthodes à partir du fichier DOT
    class_names = re.findall(r"class\s+(\w+)\s+\{", dot_string)
    method_names = re.findall(r"\blabel\s+=\s+\"(\w+)\\n", dot_string)

    # Extraire les commentaires du code source
    comments = re.findall(r'""".+?"""', code_string, re.DOTALL)

    # Créer une liste de dictionnaires contenant les informations sur chaque méthode de chaque classe
    method_info = []
    for class_name in class_names:
        class_methods = []
        for method_name in method_names:
            if method_name.startswith(class_name):
                method_docstring = ""
                for comment in comments:
                    if method_name in comment:
                        method_docstring = comment.strip('"""').strip()
                        break
                class_methods.append({
                    "name": method_name,
                    "docstring": method_docstring
                })
        method_info.append({
            "class_name": class_name,
            "methods": class_methods
        })

    # Générer le prompt à partir des informations extraites
    prompt = ""
    for class_info in method_info:
        prompt += f"La classe {class_info['class_name']} a les méthodes suivantes :\n"
        for method in class_info["methods"]:
            prompt += f"- {method['name']}: {method['docstring']}\n"
        prompt += "\n"

    return prompt


class commentateur:
    def __init__(self, path_to_watch=None, path_to_save=None, path_to_copy=None, watchdog: bool = False):
        """
        This program allow to auto comment py file with gpt3.5 by pushing file in a folder
        :param path_to_watch: Waiting a new file
        :param path_to_save: Retrieve your commented py file
        :param path_to_copy: Make a copy of an original file
        """
        if path_to_watch is None:
            path_to_watch = "./Push_code_here"
        if path_to_save is None:
            path_to_save = "./Modified"
        if path_to_copy is None:
            path_to_copy = "./Original"

        self.print = []

        if watchdog:
            # Vérifiez si le dossier "Push code here" existe
            self.push_code_here_path = path_to_watch
            if not os.path.exists(self.push_code_here_path):
                self._print(f"The Folder {path_to_watch} doesn't exist")
                os.mkdir(self.push_code_here_path)
                self._print(f"Folder {path_to_watch} created")


            # Vérifiez si le dossier "Original" existe, sinon le créez
            self.original_path = path_to_copy
            if not os.path.exists(self.original_path):
                os.mkdir(self.original_path)
                self._print(f"Folder {path_to_copy} created")

            # Vérifiez si le dossier "Modified" existe, sinon le créez
            self.modified_path = path_to_save
            if not os.path.exists(self.modified_path):
                os.mkdir(self.modified_path)
                self._print(f"Folder {path_to_save} created")

    def process_folder(self):
        while True:
            for dir_path, dir_names, file_names in os.walk(self.push_code_here_path):
                for dir_name in dir_names:
                    orig_dir_path = os.path.join(dir_path, dir_name).replace(self.push_code_here_path,
                                                                             self.original_path)
                    mod_dir_path = os.path.join(dir_path, dir_name).replace(self.push_code_here_path,
                                                                            self.modified_path)

                    os.makedirs(orig_dir_path, exist_ok=True)
                    os.makedirs(mod_dir_path, exist_ok=True)

                    dir_util.copy_tree(os.path.join(dir_path, dir_name), orig_dir_path)

                for filename in file_names:
                    file_path = os.path.join(dir_path, filename)

                    orig_path = file_path.replace(self.push_code_here_path, self.original_path)
                    mod_path = file_path.replace(self.push_code_here_path, self.modified_path)

                    os.makedirs(os.path.dirname(orig_path), exist_ok=True)
                    os.makedirs(os.path.dirname(mod_path), exist_ok=True)

                    shutil.copyfile(file_path, orig_path)

                    self.compute_file(file_path, mod_path)

                    os.remove(file_path)

            for root, dirs, _ in os.walk(self.push_code_here_path, topdown=False):
                for name in dirs:
                    try:
                        os.rmdir(os.path.join(root, name))
                    except OSError:
                        pass  # If the directory is not empty, an OSError is raised, in that case

            time.sleep(1)

    def compute_file(self, orig_filepath, dest_filepath):
        item = os.path.basename(orig_filepath)  # get the file name not the path
        item_path = orig_filepath

        _, file_extension = os.path.splitext(item_path)

        if file_extension == ".py":
            self._print(f"Working on {item}")
            with open(item_path, "r") as file:
                code = file.read()
            modified_code, short_resume = self.add_python_docstring(code)
            modified_code = autopep8.fix_code(modified_code)

            with open(dest_filepath, "w") as file:
                file.write(modified_code)

            self.comment_full_code(dest_filepath, short_resume)

            self.correct_py_file(dest_filepath)
        else:
            shutil.copyfile(orig_filepath, dest_filepath)

    def _TO_IMPLEMENT(self):

        # C part  NOT IMPLEMENTED YET
        if file_extension == ".c":
            # Read the file
            with open(item_path, "r") as file:
                code = file.read()
            modified_code = self.comment_c_function(code)
            # Write the new code to the file
            with open(os.path.join(path_modified, item), "w") as file:
                file.write(modified_code)

        elif file_extension == ".mat":
            print()
            # extract_matlab_function(item_path)
            # extract_matlab_method(item_path)

        elif file_extension == ".pas" or file_extension == ".p":
            # Read the file
            with open(item_path, "r") as file:
                code = file.read()
            modified_code = extract_pascal_functions(code)
            # Write the new code to the file
            with open(os.path.join(path_modified, item), "w") as file:
                file.write(modified_code)
            python_convertion = input(
                "Do you want to converting pascal code functions to python ? (y/n)")
            if python_convertion == "y":
                # python_code = convert_turbo_to_python(modified_code)
                python_code = ""
                # Write the new code to the file
                file, ext = os.path.splitext(item)
                file = file + ".py"
                with open(os.path.join(path_modified, file), "w") as file:
                    file.write(python_code)
                self._print("File " + item + " converted in python -> " + str(file))

    def arg_usage(self, path_file):
        dir, file = os.path.split(path_file)
        _, file_extension = os.path.splitext(file)

        shutil.copy(path_file, os.path.join(dir, '_' + file))
        self._print("Original file copied with _ before.")
        if file_extension == ".py":
            # Read the file
            with open(path_file, "r") as file:
                code = file.read()
            # modified_code = comment_python_functions(code)
            modified_code, short_resume = self.add_python_docstring(code)
            modified_code = autopep8.fix_code(modified_code)
            # Write the new code to the file
            with open(path_file, "w") as file:
                file.write(modified_code)

            # commente le code complet :
            self.comment_full_code(path_file, short_resume)
            # Vérifie et commente d'eventuelle lignes non commentées
            self.correct_py_file(path_file)

    def add_python_docstring(self, code_str):
        """This function adds detailed python docstrings to functions in a given code string.

        Parameters:
        - code_str (str): A string containing python code.

        Returns:
        - new_code_str (str): A string containing the updated python code with docstrings.
        - resume_all_docstring (str): A string containing a summary of the docstrings added to the functions.

        The function first extracts all the functions in the code string using the 'extract_functions' function.
        It then retrieves the name of each function using the 'noms_fonctions_dans_code' function and compares it to the list of function names already present in the code string.
        If the name of a function is not present, it is skipped. Otherwise, the function generates a docstring using GPT-3 and indents it properly before inserting it at the start of the function's code block.
        The updated code string is returned along with a summary of the docstrings added to the functions."""
        resume_all_docstring = ""
        new_code_str = code_str
        functions = self.extract_functions(code_str)
        functions_names = self.noms_fonctions_dans_code(code_str)
        nb_func = len(functions)
        for function_str, start_index in functions:
            print("Function untraited : " + str(nb_func))
            function_name = self.noms_fonctions_dans_code(function_str)
            if function_name[0] not in functions_names:
                nb_func -= 1
                continue
            else:
                functions_names.remove(function_name[0])

            # Récupère le docstring
            doc_string = self.GPT_choice("Turbo", "docstring google style python", function_str)
            short_docstring = self.GPT_choice("Turbo", "short docstring", doc_string)

            # Supprime les chevrons éventuels
            doc_string = "\n".join([ligne.replace(">>>", "") for ligne in doc_string.split("\n")])
            short_docstring = "\n".join([ligne.replace(">>>", "") for ligne in short_docstring.split("\n")])
            resume_all_docstring += function_name[0] + \
                                    " : " + short_docstring + "\n"
            doc_string = self.verify_triple_quotes(doc_string)

            # print(doc_string)

            # Indente le docstring en utilisant la même indentation que la fonction d'origine
            indentation = self.get_indentation(function_str)
            doc_string = self.indent_code_str(doc_string, len(indentation) + 4)

            lines = function_str.split('\n')
            if len(lines) > 1:
                lines.insert(1, doc_string)
                code_str = "\n".join(lines)
            else:
                code_str = function_str
            new_code_str = new_code_str.replace(function_str, code_str)
            nb_func -= 1

        return new_code_str, resume_all_docstring

    @staticmethod
    def verify_triple_quotes(s):

        if not s.startswith("\"\"\""):
            s = "\"\"\"" + s
        if not s.strip().endswith("\"\"\""):
            s = s + "\"\"\""

        # Trouve l'index de la première et la dernière occurrence de '"""'
        first_index = s.find('"""')
        last_index = s.rfind('"""')

        # Si '"""' ne se trouve pas dans la chaîne, retourne la chaîne inchangée
        if not (first_index == -1 or last_index == -1):
            # Remplace toutes les occurrences de '"""' par "'''" sauf pour la première et la dernière occurrence
            s = s[:first_index + 3] + s[first_index + 3:last_index].replace('"""', "'''") + s[last_index:]

        return s

    def noms_fonctions_dans_code(self, code_str):
        """
        noms_fonctions_dans_code(code_str)

        Fonction qui prend en paramètre un string représentant du code Python.
        La Fonction parse le code avec le module AST de Python et analyse l'arbre syntaxique avec la fonction ast.walk.
         Elle retourne la liste contenant les noms des fonctions du code Python.

        Paramètres:
        - code_str (str) : Une string représentant du code Python.

        Retourne:
        - noms_fonctions (list) : Une liste contenant les noms des fonctions du code Python.
        """
        # déindente si necessaire
        indent = len(self.get_indentation(code_str))
        if indent > 0:
            code_str = self.indent_code_str(code_str, -indent)
        # Analyser le code source avec le module AST de Python
        arbre_syntaxe = ast.parse(code_str)

        # Parcourir l'arbre syntaxique pour trouver toutes les définitions de fonctions
        noms_fonctions = [n.name for n in ast.walk(
            arbre_syntaxe) if isinstance(n, ast.FunctionDef)]

        # Retourner la liste des noms de fonctions
        return noms_fonctions

    @staticmethod
    def get_dependencies(code_str):
        """
        Takes in a string of Python code and returns a list of its module-level dependencies.

        Parameters:
        - code_str (str): A string of Python code containing import statements

        Returns:
        - A list of all module-level dependencies found in the input code string

        A module-level dependency is defined as a module or package that is imported in the code string. This function uses Python's built-in ast (Abstract Syntax Tree) module to parse the input code string and find all import statements. It iterates through each node in the tree and checks if it is an Import or ImportFrom node. If it is an Import node, it adds each module or package name to a set of dependencies. If it is an ImportFrom node, it checks the module name and adds it to the set if it is not empty. Finally, the set of dependencies is returned as a sorted list.
        """
        dependencies = set()

        tree = ast.parse(code_str)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dependencies.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module
                if module_name:
                    dependencies.add(module_name)

        return list(dependencies)

    def comment_python_functions(self, code_str):
        """
        This function takes a string representation of Python code as input and returns a modified string after modifying the functions found in the code using GPT_choice function. The modifications are made by re-indenting the modified function with the same indentation level as the original function.

        Parameters:
        - code_str: A string representing Python code.

        Returns:
        - new_code_str: A modified string after modifying the functions found in the code using GPT_choice function. The modifications are made by re-indenting the modified function with the same indentation level as the original function.
        """
        new_code_str = code_str
        # functions = extract_python_fonction(code_str)
        functions = self.extract_functions(code_str)

        nb_func = len(functions)
        for function_str, start_index in functions:
            # debug = function_str
            print("Function untraited : " + str(nb_func))
            # Récupère la fonction modifiée
            modified_function_str = self.GPT_choice("Turbo", "Add Python", function_str)
            # modified_function_str = debug
            # Ré-indente la fonction modifiée en utilisant la même indentation que la fonction d'origine
            indentation = self.get_indentation(function_str)
            modified_function_str = self.indent_code_str(
                modified_function_str, len(indentation))
            new_code_str = new_code_str.replace(
                function_str, modified_function_str)
            nb_func -= 1

        return new_code_str

    @staticmethod
    def indent_code_str(code_str, indentation=4):
        """
        This function takes a string of code and an integer representing the indentation level
        and returns the formatted code string with appropriate indentation.

        Parameters:
        - code_str (string): the string of code to be indented or de-indented
        - indentation (int): the number of spaces to indent/de-indent the code. Positive values indent, negative values de-indent.

        Returns:
        - formatted_code (string): the formatted code string with appropriate indentation applied
        """
        # diviser le code en une liste de lignes
        lines = code_str.splitlines()
        # vérifier si on doit indenter ou désindenter
        if indentation > 0:
            # indenter le code
            for i in range(len(lines)):
                lines[i] = ' ' * indentation + lines[i]
        elif indentation < 0:
            # désindenter le code
            for i in range(len(lines)):
                lines[i] = lines[i][abs(indentation):]
        # reformer le code en une chaîne de caractères
        formatted_code = '\n'.join(lines)
        return formatted_code

    @staticmethod
    def get_indentation(code_str):
        """
        This function takes in a string that represents a block of code as input.
        It looks for the first non-empty line of the code block and returns the indentation style used by that line.
        If the input code string is empty, it returns an empty string.

        Args:
            code_str (str): The string representing the code block.

        Returns:
            str: The indentation style used by the first non-empty line of code, which could contain spaces, tabs or both.
                 If there is no indentation on the first non-empty line or if the input string is empty, it returns an empty string.
        """
        # Trouve la première ligne non vide
        for line in code_str.split('\n'):
            if line.strip():
                break
        else:
            # Si la chaîne est vide, retourne une chaîne vide
            return ''

        # Retourne les espaces ou les tabulations de l'indentation de la première ligne
        indentation = ''
        for char in line:
            if char in (' ', '\t'):
                indentation += char
            else:
                break
        return indentation

    @staticmethod
    def format_langage(langage):
        """Function to format a given programming language

            This function takes a parameter langage (string) and based on its input value, it returns a formatted dictionary
            containing various attributes such as the type of programming language, start and stop syntax of a function, 
            commenting syntax and prompt to provide guidelines to the user to write a docstring for the function in the said language.

            Args:
                langage (string): Input programming language for which the format is to be determined.

            Returns:
                Dictionary: A formatted dictionary containing language type, start and stop syntax of a function, commenting 
                syntax and prompt to provide guidelines to the user to write a docstring in the said language.

            Raises:
                ValueError: When the langage parameter is unsupported.

            Examples:
                format_langage('docstring python')
                {'langue': 'Python 3.7', 'com1': '#', 'com2': '"""', '

        if langage.lower() == "docstring python":
            formated = {"langue": "Python 3.7", "com1": "#", "com2": "\"\"\"", "start": "def ",
                        "prompt": "# Convert the above function respecting PEP 7 and PEP 257 convention:\n",
                        "role": "You must write the detailed python docstring following PEP 257 of the function below as a python comment starting with \"\"\" and ending with \"\"\"",
                        "stop": ["def"], "engine": "Turbo"}

        elif langage.lower() == "short docstring":
            formated = {"langue": "Python 3.7", "com1": "#", "com2": "\"\"\"", "start": "def ",
                        "prompt": "# Convert the above function respecting PEP 7 and Google style convention:\n",
                        "role": "You must summary in 10 words the python docstring below", "stop": ["def"],
                        "engine": "Turbo"}

        elif langage.lower() == "add python":
            formated = {"langue": "Python 3.7", "com1": "#", "com2": "\"\"\"", "start": "def ",
                        "prompt": "# Convert the above function respecting PEP 7 and google style convention:\n",
                        "role": API_langage['en'], "stop": ["def"], "engine": "Turbo"}

        elif langage.lower() == "c":
            formated = {"langue": "c", "com1": "/* ", "com2": "*/", "start": "\"\"\"",
                        "prompt": "# An elaborate, high quality docstring for the above c function:\n",
                        "role": API_langage['en'], "stop": ["\"\"\"", "#"], "engine": "code-davinci-002"}

        elif langage.lower() == "python full code":
            formated = {"langue": "Python 3.7", "com1": "'''", "com2": "'''", "start": "\"\"\"",
                        "prompt": "# An elaborate, high quality docstring for the above c function:\n",
                        "role": "You are an expert in python programming, you must determine the usefulness of this program. your result must be a python comment",
                        "stop": ["\"\"\"", "#"], "engine": "code-davinci-002"}

        elif langage.lower() == "docstring google style python":
            formated = {"langue": "Python 3.7", "com1": "#", "com2": "\"\"\"", "start": "def ",
                        "prompt": "# Convert the above function respecting PEP 7 and Google style convention:\n",
                        "role": "You must write a python docstring following Google style python convention of the function below as a python comment starting with \"\"\" and ending with \"\"\"",
                        "stop": ["def"], "engine": "Turbo"}
        else:
            raise ValueError("Langage non pris en charge")
        return formated

    def GPT_choice(self, engine: str = "Turbo", langage: str = "Python", function_or_method: str = ""):
        """
        This function takes in three parameters:
        - engine (str): the GPT engine to use. Possible values are "Turbo", "text-davinci-003" and "code-davinci-002". Default is "Turbo".
        - langage (str): the programming language in which the function or method is written. Default is "Python".
        - function_or_method (str): the name of the function or method for which we want to generate the docstring. Default is an empty string.

        The function returns a Python string that is a detailed docstring for the specified function/method. It uses OpenAI's GPT-3 text generation API to generate the docstring.

        The function first chooses the GPT engine to use based on the engine parameter. If the engine is "Turbo", it calls the GPT_turbo function with the specified langage and function_or_method parameters. If the engine is "text-davinci-003" or "code-davinci-002", it calls the GPT_classic function with the same parameters. The result of the chosen function is returned as the final output.

        Note that the GPT_turbo and GPT_classic functions are not defined in this script and must be imported from elsewhere.
        """
        function = None
        if engine == "Turbo":
            function = self.GPT_turbo(langage, function_or_method)
        elif engine == "text-davinci-003":
            function = self.GPT_classic(langage, function_or_method)
        elif engine == "code-davinci-002":
            function = self.GPT_classic(langage, function_or_method)
        return function

    def GPT_classic(self, langage, function_or_method):
        """
        GPT_classic - Uses OpenAI GPT to generate a python docstring for the given programming language and function/method name.

        @langage: The programming language of the code snippet for which the docstring is to be generated (string)

        @function_or_method: The name of the function/method for which the docstring is to be generated (string)

        The function first formats the given language for OpenAI's GPT engine.
        It then generates a prompt for the GPT engine including the function/method name and the formatted language.
        The function then waits for user input.
        After the user enters a blank input, the function sends the prompt to OpenAI's GPT engine.
        The GPT engine generates a response as a continuation of the prompt.
        The function formats the response to fit as a python docstring.
        The final generated docstring is returned by the function.

        @return: A generated python docstring (string)
        """

        f = self.format_langage(langage)
        prompt = function_or_method + "\n" + f["prompt"] + "\n" + f["start"]
        print(prompt)
        input()
        response = openai.Completion.create(
            model=f["engine"],
            prompt=prompt,
            temperature=0.7,
            max_tokens=3500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=f["stop"]
        )
        docstring = f["com1"] + response.choices[0].text.strip() + f["com2"]

        return docstring

    def GPT_turbo(self, langage, function_or_method):
        """
        GPT_turbo is a function that takes in two parameters - langage and function_or_method. It uses OpenAI's GPT-3 natural language processing model to generate code based on the input provided.

        Parameters:
        langage (str): The programming language for which code is to be generated.
        function_or_method (str): The function or method name that needs to be generated.

        Returns:
        clean_function (str): The generated function or method as a string.

        The function first formats the input language into a standard format suitable for GPT-3 processing. It then uses the ChatCompletion.create() function of OpenAI's API to generate code based on the messages passed (role: system and user). The generated code is cleaned before being returned as a string.

        If any error occurs during the execution of the function, depending on the type of error, either an error message is printed or the function returns None.
        """

        f = self.format_langage(langage)
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-0613",
                    messages=[
                        {"role": "system", "content": f["role"]},
                        {"role": "user", "content": function_or_method}
                    ]
                )
                function = response['choices'][0]['message']['content']
                # Diviser la chaîne en plusieurs lignes
                lignes = function.split('\n')

                # Vérifier si la première ligne contient le symbole ```
                if '```' in lignes[0]:
                    # Si c'est le cas, supprimer la première ligne
                    lignes = lignes[1:]
                # Vérifier la dernière ligne
                if '```' in lignes[-1]:
                    # Si c'est le cas, supprimer la dernière ligne
                    lignes = lignes[:-1]
                # Joindre les lignes restantes
                clean_function = '\n'.join(lignes)

                return clean_function

            except OpenAIError as error:
                if error.__class__.__name__ == 'AuthenticationError':
                    print("Erreur d'authentification: vérifiez votre clé API.")
                    break
                elif error.__class__.__name__ == 'RateLimitError':
                    print("Erreur de taux de requête: Attente de 5 minutes.")
                    # Attendre 5 minutes (300 secondes) avant de réessayer
                    time.sleep(300)
                elif error.__class__.__name__ == 'APIError':
                    print("Erreur de l'API OpenAI: {}".format(error))
                    break
                else:
                    print("Une erreur s'est produite: {}".format(error))
                    time.sleep(5)

        return None

    @staticmethod
    def extract_functions(code):
        """
        The function extracts all functions in a given source code string and returns them as a list
        of tuples, where each tuple contains the function's source code (as a string) and its indent level (as an integer).

        Parameters:
        code (str): A string of source code containing one or more functions.

        Returns:
        list: A list of tuples, where each tuple contains the source code of a function and its indent level.

        Examples:
         code = "def foo():\n    print('Hello, world!')\n\n\nclass Bar:\n    def baz(self):\n        print('Goodbye, world!')"
         extract_functions(code)
        [("def foo():\n    print('Hello, world!')", 0), ("    def baz(self):\n        print('Goodbye, world!')", 4)]
        """
        functions = []
        current_function = None
        indent_level = 0
        for line in code.split('\n'):
            actual_indent_line = len(line) - len(line.lstrip())
            if line.strip().startswith('def '):
                if current_function:
                    functions.append((current_function, indent_level))
                current_function = line
                indent_level = len(line) - len(line.lstrip())
            elif current_function is not None:
                if line.isspace() or len(line.strip()) > 0:
                    if actual_indent_line > indent_level:
                        current_function += '\n' + line
                    else:
                        functions.append((current_function, indent_level))
                        current_function = None
                        indent_level = 0
                else:
                    current_function += '\n' + line
        if current_function:
            functions.append((current_function, indent_level))

        return functions

    @staticmethod
    def extract_C_functions(content):
        """
        Extracts C functions from input code content string and returns a list of extracted functions.

        Args:
            content (str): A string containing C code from which functions have to be extracted.

        Returns:
            list: A list of all extracted C functions.

        Raises:
            None.
        """

        pattern = r'\b(\w+)\s+(\w+)\s*\([^)]*\)\s*{'

        functions = []
        start_index = 0

        for match in re.finditer(pattern, content):
            # Exclude blocks of code containing a newline before the opening brace
            if not '\n{' in content[match.start():match.end()]:
                continue
            # Find the start of the function body
            start_index = match.end()

            # Find the end of the function body
            brace_count = 1
            end_index = start_index
            while brace_count > 0:
                if content[end_index] == '{':
                    brace_count += 1
                elif content[end_index] == '}':
                    brace_count -= 1
                end_index += 1

            # Add the function to the list
            function = content[match.start():end_index]
            functions.append(function)

        return functions

    def comment_c_function(self, code_str):
        """
        This function takes a C code as a string and adds a docstring to each function present in the code. It uses the function `extract_C_functions` to extract all the functions defined in the code. Then, it iterates over the list of functions, optimizes them using `optimize_token_c`, generates a docstring using `GPT_choice` (which takes an openAI GPT API key, language type and text as input), and adds the docstring to the second line of each function block by splitting and joining the function string. Finally, it replaces the old function string with the new modified string in the code string, and returns the modified code string with added docstrings.

        Parameters:
        - code_str: A string containing the C code to which docstrings need to be added.

        Returns:
        - new_code_str: A modified string containing the C code with added docstrings to each function present in the code.
        """
        new_code_str = code_str
        functions = self.extract_C_functions(code_str)
        nb_func = len(functions)
        for function in functions:
            print("Function untraited : " + str(nb_func))
            # opt_function = optimize_token_c(function)
            opt_function = function
            docstring = self.GPT_choice("code-davinci-002", "C", opt_function)
            # Diviser la chaîne en plusieurs lignes
            lines = function.split('\n')
            # Ajouter la nouvelle chaîne à la deuxième ligne
            lines[1] += "/* " + docstring + "\n*/"
            modified_function_str = '\n'.join(lines)

            new_code_str = new_code_str.replace(function, modified_function_str)
            nb_func -= 1
        return new_code_str

    @staticmethod
    def optimize_token_c(function):
        """
        This function takes in a function code as a string and performs the following optimization:
        1. Removes any leading white spaces from each line of code
        2. Joins the modified lines to reform a single string of code
        It then returns the optimized code string.

        Parameters:
        function (str): A string containing the code of a Python function

        Returns:
        str: The optimized code of the input function
        """
        lignes = function.split("\n")
        for i in range(len(lignes)):
            lignes[i] = lignes[i].lstrip()
        # Joindre les lignes modifiées pour reformer la chaîne
        chaine_modifiee = '\n'.join(lignes)
        return chaine_modifiee

    @staticmethod
    def check_compile(filename):
        """
        check_compile: Function to check if the code present in the given filename compiles correctly by using ast module in python.

        @filename : file name (with complete path) of the source code file to check compile of.

        Return: None.

        The function reads the source code from the file present at location specified by the given filename.
        It then parses the source code using ast.parse() method and also does necessary modifications using ast.RemplacerEmptyValue().visit(tree).
        It then walks through all nodes present in ast tree and tries to compile those nodes which are type of ast.Expr and also have value attribute.
        If the compilation fails, it logs the error message in the error_log.txt file and also updates the code by adding "#" at the position where compilation error was found.
        """
        with open(filename, "rt") as file:
            source = file.read()

        tree = ast.parse(source)
        tree = ReplaceEmptyValue().visit(tree)

        for node in ast.walk(tree):
            print(node)
            if isinstance(node, ast.AST) and isinstance(node, ast.Expr) and hasattr(node, "value"):
                try:
                    compile(ast.Expression(node), filename, "eval")
                except SyntaxError as error:
                    print(f"{error.__class__.__name__}: {error}")
                    with open(filename, "rt") as file:
                        lines = file.readlines()
                    col_offset = error.offset - \
                                 sum(len(line) for line in lines[:error.lineno - 1])
                    line = lines[error.lineno - 1].rstrip()
                    lines[error.lineno -
                          1] = f"{line[:col_offset]}# {line[col_offset:]}\n"
                    with open(filename, "wt") as file:
                        file.write("".join(lines))
                    with open("error_log.txt", "at") as file:
                        file.write(
                            f"{filename}:{error.lineno}:{col_offset}: {error.__class__.__name__}: {error}\n")

    @staticmethod
    def compile_py_file(file_name):
        """
        Takes in a file name and compiles it as a python file.

        Parameters:
        file_name (str): The name of the python file to be compiled

        Returns:
        int: The line number of the error occurred during compilation, if any

        Raises:
        None

        Example:
        Suppose the python file named `myfile.py` contains the following code:
        print("Hello, World!")

        The function can be called as:
        compile_py_file('myfile.py')

        The function will compile the code in `myfile.py` and return None if successful. Otherwise, if there is an error during compilation, it will return the line number where the error occurred as an integer.
        """
        try:
            compile(open(file_name).read(), file_name, 'exec')
        except Exception as e:
            return e.lineno

    def comment_error_line(self, file_name):
        """
        Function name: comment_error_line

        Parameters:
        - file_name (str): a string representing the name of the Python file to be compiled

        Returns:
        - None

        Description:
        - This function takes in the name of a Python file, compiles it and checks if there is an error.
        - If there is an error, it comments out the line containing the error by adding a "#" symbol at the beginning of the line.
        - The function then updates the original file with the appropriately commented lines.
        """
        error_line = self.compile_py_file(file_name)
        if error_line:
            with open(file_name, 'r') as f:
                lines = f.readlines()
            lines[error_line - 1] = '#' + lines[error_line - 1]
            with open(file_name, 'w') as f:
                f.writelines(lines)

    @staticmethod
    def open_py_file(file_name):
        """
        This function takes a filename as a parameter and returns the contents of the file as a string.

        file_name: A string representing the name of the file to be opened.

        Returns a string containing the contents of the file.

        The function uses the 'with' statement to open the file in read mode. It then reads the contents of the file using the 'read()' method and returns the contents as a string.

        If the file cannot be opened, the function will raise an exception.
        """
        with open(file_name, "r") as file:
            return file.read()

    @staticmethod
    def compile_py_code(code):
        '''
        This function takes a string argument 'code' and attempts to compile it using the built-in Python function: compile. The code is compiled as an executable by specifying the third argument as "exec" in the compile function. If there is a syntax error, the function catches it using a try-except block and returns the line number where the error occurred as an integer.

        Args:
            code (str): A string containing the Python code to be compiled.

        Returns:
            (int): The line number where the SyntaxError occurred, if any. If there are no syntax errors, 'None' is returned.

        Example:
            compile_py_code('print("Hello World!")')
            None

            compile_py_code('print("Hello World!"')
            1
        '''
        try:
            compile(code, "", "exec")
        except SyntaxError as e:
            return e.lineno

    def correct_py_file(self, file_name, code: str = ""):
        '''
        This function corrects a python file by identifying and commenting out any syntax errors in the file.

        Arguments:
        - file_name (str): the name of the file to be corrected

        Returns:
        - None

        The function takes in a file name and opens the file using the `open_py_file()` function. Then, in a while loop, it compiles the code using the `compile_py_code()` function to check for any syntax errors. If an error is found, it identifies the line of the error and comments out that line by adding a `#` at the beginning of the line. The loop repeats until there are no more syntax errors in the code.

        Finally, the corrected code is written back to the original file using the `with open()` statement and the write mode. The function does not return anything, it only modifies the file.

        IMPORTANT NOTE: The `open_py_file()` and `compile_py_code()` functions used within this function are not defined within the code provided, so this function cannot be run as is. Those functions need to be provided or defined first.
        '''
        if file_name:
            code = self.open_py_file(file_name)
        while True:
            line_error = self.compile_py_code(code)
            if line_error is None:
                break
            code = code.split("\n")
            code[line_error - 1] = "#" + code[line_error - 1]
            code = "\n".join(code)
        if file_name:
            with open(file_name, "w") as file:
                file.write(code)
            return
        else:
            return code

    # ______________________________________________________________________
    # Comment full program

    def extract_return_var_names(self, child_node):
        '''
        Extracts return variable names from an Abstract Syntax Tree (AST) node of Python code.

        Parameters:
            child_node (ast.AST): An AST node of Python code.

        Returns:
            A list of strings representing the names of the return variables extracted from the child node.

        Raises:
            None

        Example:
            ast_node = ast.parse("def add(x, y):\n    return x+y")
            return_vars = extract_return_var_names(ast_node.body[0])
            print(return_vars) # Output: ['x', 'y']
        '''
        var_names = []
        if isinstance(child_node, ast.Return):
            # Si le nœud est de type "Return", extraire les noms des variables de retour
            if isinstance(child_node.value, ast.Tuple):
                for elt in child_node.value.elts:
                    var_names += self.extract_return_var_names(elt)
            elif isinstance(child_node.value, ast.Name):
                var_names.append(child_node.value.id)
        elif isinstance(child_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Si le nœud est de type "FunctionDef", "AsyncFunctionDef" ou "ClassDef", ne pas descendre dans l'arbre
            return []
        else:
            # Sinon, descendre dans l'arbre et continuer à chercher les noms de variables de retour
            for sub_node in ast.iter_child_nodes(child_node):
                var_names += self.extract_return_var_names(sub_node)
        return var_names

    def comment_full_code(self, file_path, short_resume):
        '''This function extracts each function or method of a given file_path in the format
        function_name(arg1, arg2) --> return None | val1 |val2, concatenates them into a string, generates a
        comment and adds it at the beginning of the file. It takes two arguments:
                - file_path: the path of the file to extract the functions and methods from
                - short_resume: a summary of the functions and methods to include in the comment

        The function does the following:
            - Reads the contents of file_path
            - Extracts the functions, methods and classes from the file using Abstract Syntax Trees (AST)
            - Extracts the name and arguments for each function/method and the name of the class if present
            - Extracts the names of the variables returned by each function/method
            - Builds a string representation of this information
            - Extracts the package/module dependencies of the code using get_dependencies function
            - Builds a string representation of the imports to add to the code
            - Concatenates this information into one string
            - Calls the function GPT_choice with the parameters "Turbo", "Python full code", and
            the concatenated string as the argument to generate a comment using GPT model
            - Prepends this comment to the contents of the file_path and writes it back to the file
            - Returns the generated comment'''
        # Cette fonction extrait chaque fonction ou méthode au format function_name(arg1, arg2) --> return None | val1 |val2
        # Concatène l'ensemble en string, puis génère un commentaire qui sera placé au début du fichier file_path
        with open(file_path, "r") as f:
            file_contents = f.read()

        parsed_tree = ast.parse(file_contents)

        functions_and_methods = []
        for node in ast.walk(parsed_tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Extraire le nom de la fonction ou de la méthode
                name = node.name

                # Extraire les arguments de la fonction ou de la méthode
                args = []
                for arg_node in node.args.args:
                    args.append(arg_node.arg)

                # Extraire les noms des variables de retour
                var_names = []
                for child_node in ast.iter_child_nodes(node):
                    var_names += self.extract_return_var_names(child_node)

                # Ajouter l'information dans la liste des fonctions et méthodes
                info_str = f"{name}({', '.join(args)}) --> {', '.join(var_names)}"
                functions_and_methods.append(info_str)

            elif isinstance(node, ast.ClassDef):
                # Extraire le nom de la classe
                name = node.name

                # Ajouter l'information dans la liste des classes
                info_str = f"class {name}"
                functions_and_methods.append(info_str)
        dependence = self.get_dependencies(file_contents)
        resume = '\n'.join(functions_and_methods)
        resume_code = ""
        for dep in dependence:
            resume_code += "import " + dep + "\n"
        resume_code += resume + "\n" + "Summary of function : \n" + short_resume
        reponse = self.GPT_choice("Turbo", "Python full code", resume_code)

        with open(file_path, 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write(reponse.strip() + "\n" + content)
        return reponse

    def comment_unique_fonction(self, code):
        # modified_code = comment_python_functions(code)
        modified_code, short_resume = self.add_python_docstring(code)
        modified_code = autopep8.fix_code(modified_code)
        # Write the new code to the file

        # Vérifie et commente d'eventuelle lignes non commentées
        code_with_docstring = self.correct_py_file("", modified_code)
        return code_with_docstring, short_resume

    def _print(self, text, role: str = "system_print"):
        print(text)
        self.print.append({role: text})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="File to comment (other parameters not necessary")
    parser.add_argument("-o", "--original", help="Path folder to copy original file before comment (for infinite loop)")
    parser.add_argument("-m", "--modified", help="Path folder to with commented file (for infinite loop)")
    parser.add_argument("-p", "--push", help="Path folder waiting a new file or folder to comment (for infinite loop)")
    args = parser.parse_args()

    if get_openai_api_key() is not None:
        if args.file:
            comment = commentateur(watchdog=False)
            comment.arg_usage(args.file)
        else:
            ptw = args.push if args.push else None
            ptc = args.original if args.original else None
            pts = args.modified if args.modified else None
            comment = commentateur(path_to_watch=ptw, path_to_copy=ptc, path_to_save=pts, watchdog=True)
            comment.process_folder()
