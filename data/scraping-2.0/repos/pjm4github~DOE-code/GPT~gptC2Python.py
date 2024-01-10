import re
import codecs
import os
import time
from openai import OpenAI

# Replace 'YOUR_API_KEY' (as an ENV variable) with your actual GPT-3 API key


import urllib.parse
import re


def remove_multiline_comments(code):
    """
    Removes all multiline comments from the given code.
    all block comments (enclosed by /* and */) from a C++ file are removed
    """
    cleaned_code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

    # Remove all multiline comments from the code.
    return cleaned_code


def remove_single_line_comments(input_code):
    """
    Removes all block comments from Java code.
    block comments are start with //
    Args:
      code: The Java code to remove comments from.

    Returns:
      The Java code with all comments removed.
    """

    # Use regular expression to remove // comments
    cleaned_code = re.sub(r'//.*', '', input_code)

    # Remove all block comments from the code.
    return cleaned_code


class GptCodeConverter():

    MODEL_CHOICE_1 = "gpt-3.5-turbo-1106"
    MODEL_CHOICE_2 = "code-davinci-002",
    MODEL_CHOICE_3 = "gpt-3.5-turbo",
    # max_tokens=500,  # Adjust as needed
    # temperature=0.7  # Adjust the temperature for creativity

    MAX_TOKENS = 10000  # Maximum number of tokens that can be used with the OPENAI model (model dependant)

    def __init__(self, language="Java", model=MODEL_CHOICE_1):
        self.client = OpenAI(
                                # defaults to os.environ.get("OPENAI_API_KEY")
                                # api_key=api_key,
                            )
        self.model_name = model
        self.language = language
        self.setup_instructions = f"Given this {language} code class convert it to python using snake_case methods names. Keep the class names in CamelCase."
        self.add_function_instructions = f"Given this {language} function function convert it to python using snake_case function names."
        self.add_class_instructions = f"Given this class convert that code to python using snake_case method names."

    def convert_code(self, code_snippet, instructions):
        """
        Convert the given code snippet using GPT-3.
        """
        # Call the GPT-3 API to generate the converted code
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": instructions
                    },
                    {
                        "role": "user",
                        "content": code_snippet
                    }
                ],
                model=self.model_name,

            )

            # Extract and return the generated code from the response

            converted_code = chat_completion.choices[0].message.content
        except Exception as e:
            print(e)
            converted_code = ''
        self.converted_code = converted_code


class CCodeParser:
    def __init__(self, fn=None):
        self.input_code_filename = fn
        self.classes = []
        self.functions = []
        self.c_code = ""
        self.blanked_code = ""

    def load_file(self, filename=None):

        if filename:
            self.input_code_filename = filename
        print(f"\n\n################################\nLOADING FILE {self.input_code_filename}")
        with open(self.input_code_filename, 'r') as file:
            self.c_code = file.read()
        self.blanked_code = self.c_code

    def un_load_file(self):
        self.input_code_filename = None
        self.classes = []
        self.functions = []
        self.c_code = ""
        self.blanked_code = ""

    def dump_classes(self, full=False):
        for c in self.classes:
            return_type = c["Return"] + " " if c["Return"] else ""
            s = f'{return_type}{c["Class"]}::{c["Method"]}'
            if full:
                s +=f'({c["Arguments"]})\n{{{c["Body"]}}}\n'
            print(s)

    def dump_functions(self, full=False):
        for g in self.functions:
            if g["Function"] and g["Function"] not in ['if', 'for', 'while']:  # hack
                s = f'{g["Return"]} {g["Function"]}'
                if full:
                    s += f'({g["Arguments"]})\n{{{g["Body"]}}}\n'
                print(s)

    def snarf_classes(self):
        print("SCANNING for CLASSES...")
        test = self.c_code

        pattern = r"""((?P<return_type>\w+)\s+)*(?P<class_name>\w+)::(?P<method_name>\w+)\((?P<arguments>[^)]*)\)\s*{"""
        max_len = len(self.c_code)
        # Find all matches of the pattern in the source code
        matches = re.finditer(pattern, self.c_code)
        for m in matches:
            span = m.span()
            # Now walk forward in the code and match the braces until the braces are balanced to find the end of the method body
            # test_code = self.c_code[span[2]:]
            brace_count = 1  # We start with 1 since that's already included in the span
            method_end = span[1]
            #############################
            # CLASS SCANNER
            while brace_count:
                if method_end >= max_len:
                    print(f"something went wrong with the class scanner, skipping {m.group('class_name')}::{m.group('method_name')},")
                    break
                test_char = self.c_code[method_end]
                # need to qualify the characters to make sure that they are not escaped
                if test_char == "{":
                    brace_count += 1
                elif test_char == "}":
                    brace_count -= 1
                method_end += 1
            if method_end >= max_len:
                continue

            method_body = self.c_code[span[1]: method_end-1]  # does not include the opening and closing braces

            class_dict = {"Return": m.group('return_type'),
                                 "Class": m.group('class_name'),
                                 "Method": m.group('method_name'),
                                 "Arguments": m.group('arguments'),
                                 "Body": method_body,
                                 "BodySpan": (span[1], method_end-1)}
            self.classes.append(class_dict)


        # # pattern = r"""((?P<return_type>\w+)\s+)*(?P<class_name>\w+)::(?P<method_name>\w+)\((?P<arguments>[^)]*)\)\s*{(?P<method_body>(?:[^{}]*\{[^{}]*\})*[^{}]*)}"""
        # # pattern = r"""\s*(?P<return_type>\w+)\s+(?P<class_name>\w+)::(?P<method_name>\w+)\((?P<arguments>[^)]*)\)\s*{(?P<method_body>(?:[^{}]*|{(?:[^{}]*|{(?:[^{}]*|{[^{}]*})*})*})*})"""
        # pattern = r"""((?P<return_type>\w+)\s+)*(?P<class_name>\w+)::(?P<method_name>\w+)\((?P<arguments>[^)]*)\)\s*{(?P<method_body>(?:[^{}]*|{(?:[^{}]*|{(?:[^{}]*|{[^{}]*})*})*})*)}"""
        # p_compile = re.compile(pattern, re.MULTILINE)
        # matches = p_compile.finditer(self.c_code)
        # # matches = re.finditer(pattern, self.c_code, re.MULTILINE)
        #
        # for match in matches:
        #     # For each of the matches, capture_span holds the span of the body match for that class.
        #     # This is used to postprocess the file and remove the class body to produce only a skeleton version
        #     # of the code that will be sent to Open.AI for conversion into python.
        #     # The trick is to find the body closest to the point after the class declaration because some class bodies
        #     # will match everything (e.g. and empty class body)
        #     print(f"len={len(self.c_code)}, {match.span()}, CLASS: {match.group('class_name')}::{match.group('method_name')}")
        #     capture_span = None
        #
        #     sb = re.finditer(re.escape(match.group('method_body').strip()), self.c_code, re.MULTILINE)
        #     body_spans = []
        #     for m in sb:
        #         body_spans.append(m.span())
        #
        #     se = re.finditer(re.escape(match.group('method_name') + "(" + match.group('arguments') + ")"),
        #                      self.c_code, re.MULTILINE)
        #     class_spans = []
        #     for m in se:
        #         class_spans.append(m.span())
        #     # Find the location of the body span that is closest to the end of the class_span
        #     if len(class_spans) == 1:  # Only do this if there is one matching class otherwise it may be wrong.
        #         cse = class_spans[0][1]
        #
        #         for body_span in body_spans:
        #             if body_span[0] > cse:
        #                 capture_span = body_span
        #                 break
        #     # Assemble the class structure
        #     class_dict = {"Return": match.group('return_type'),
        #                          "Class": match.group('class_name'),
        #                          "Method": match.group('method_name'),
        #                          "Arguments": match.group('arguments'),
        #                          "Body": match.group('method_body'),
        #                          "BodySpan": capture_span}
        #     self.classes.append(class_dict)
        # this is how to replace the code with spaces
        print("... DONE SCANNING for CLASSES")

        for c in self.classes:
            span = c.get("BodySpan", None)
            if span:
                start_pos = span[0]
                end_pos = span[1]
                self.blanked_code = self.blanked_code[:start_pos] + ' ' * (end_pos - start_pos) + self.blanked_code[

                                                                                              end_pos:]
    def snarf_function(self):
        print("SCANNING for FUNCTIONS ... ")

        test = self.c_code
        pattern = r"""(?P<return_type>\w+)\s+(?P<function_name>[A-Za-z0-9_*]*)\((?P<arguments>[^)]*)\)\s*{(?P<function_body>(?:[^{}]*\{[^{}]*\})*[^{}]*)}"""
        matches = re.finditer(pattern, test)
        for match in matches:
            if match.group('function_name') and match.group('function_name') in ['if', 'for', 'while']:  # hack
                break # skip over the if, for and while statemenst that are captured by the regexp pattern above
            print(f"len={len(self.c_code)}, {match.span()}, FUNCTION: {match.group('function_name')}")
            capture_span = None
            sb = re.finditer(re.escape(match.group('function_body').strip()), self.c_code, re.MULTILINE)
            body_spans = []
            for m in sb:
                body_spans.append(m.span())
            se = re.finditer(re.escape(match.group('function_name') + "(" + match.group('arguments') + ")"),
                             self.c_code, re.MULTILINE)
            function_spans = []
            for m in se:
                function_spans.append(m.span())

            # Find the location of the body span that is closest to the end of the class_span
            if len(function_spans) == 1:  # Only do this if there is one matching class otherwise it may be wrong.
                cse = function_spans[0][1]

                for body_span in body_spans:
                    if body_span[0] > cse:
                        capture_span = body_span
                        break

            self.functions.append({"Return": match.group('return_type'),
                                   "Function": match.group('function_name'),
                                   "Arguments": match.group('arguments'),
                                   "Body": match.group('function_body'),
                                   "BodySpan": capture_span})

        print("... DONE SCANNING for FUNCTIONS")
        # this is how to replace the code with spaces
        for f in self.functions:
            span = f.get("BodySpan", None)
            if span:
                start_pos = span[0]
                end_pos = span[1]
                self.blanked_code = self.blanked_code[:start_pos] + ' ' * (end_pos - start_pos) + self.blanked_code[end_pos:]

    def parse(self):
        print("Snarfing classes")
        self.snarf_classes()
        print("Snarfing functions")
        self.snarf_function()


def parse_and_convert(parser, directory_path, filename, current_time):
    converter = GptCodeConverter("CPP")

    s = parser.blanked_code
    # Get rid of all the white space that was inserted.
    s = '\n'.join(line for line in s.splitlines() if line.strip())
    print("Converting the base class")

    encoded_text = urllib.parse.quote(s)
    python_code = ""
    converter.convert_code(encoded_text, converter.setup_instructions)
    python_snip = converter.converted_code
    if python_snip:
        # get rid of the leading and trailing python quoting
        converted_code = python_snip.replace("```python", f"# Converted by an OPENAI API call using model: {converter.model_name}")
        converted_code = converted_code[:-3] if converted_code[-3:] == "```" else converted_code
        python_code += "\n\n" + converted_code
        # print(converted_code)
    else:
        print(f"{filename} blank conversion failed")

    for g in parser.functions:
        if g["Function"] is None :
            continue
        if isinstance(g["Function"], str):
            if g["Function"].strip() == "":
                print("     skipping an empty function")
                continue

        print(f'Converting a function: {g["Function"]}')
        s = f'{g["Return"]} {g["Function"]}({g["Arguments"]})\n' \
            f'{{{g["Body"]}}}\n'

        # remove comments
        if len(s) > GptCodeConverter.MAX_TOKENS:
            s = remove_single_line_comments(s)
            s = remove_multiline_comments(s)

        encoded_text = urllib.parse.quote(s)
        converter.convert_code(s, converter.add_function_instructions)
        python_snip = converter.converted_code
        if python_snip:
            # get rid of the leading and trailing python quoting
            converted_code = python_snip.replace("```python",
                                                    f"# Converted by an OPENAI API call using model: {converter.model_name}")
            converted_code = converted_code[:-3] if converted_code[-3:] == "```" else converted_code
            python_code += "\n\n" + converted_code
            # print(converted_code)
        else:
            print(f"{filename}, {g['Function']} conversion failed")

    for c in parser.classes:
        return_type = c["Return"] + " " if c["Return"] else ""
        s = f'{return_type}{c["Class"]}::{c["Method"]}({c["Arguments"]})\n' \
            f'{{{c["Body"]}}}\n'
        print(f'Converting a class: {c["Class"]}::{c["Method"]}')

        # encoded_text = urllib.parse.quote(s)
        converter.convert_code(s, converter.add_class_instructions)
        python_snip = converter.converted_code
        if python_snip:
            # get rid of the leading and trailing python quoting
            converted_code = python_snip.replace("```",
                                                    f"# Converted by an OPENAI API call using model: {converter.model_name} ")
            converted_code = converted_code[:-3] if converted_code[-3:] == "```" else converted_code
            python_code += "\n\n" + converted_code
            # print(converted_code)
        else:
            print(f"{filename} {c['Class']}::{c['Method']} conversion failed")


    file_extension = '.py'
    base_filename = filename.split(".")[0]
    # Create a unique filename by appending the timestamp to a base filename and file extension
    output_filename = f"{directory_path}{base_filename}{current_time}{file_extension}"

    with open(output_filename, 'w') as f:
        f.write(python_code)
    print(f"{output_filename} written")


def main(path, filename=None):
    # directory_path = f"{os.path.expanduser('~')}/Documents/Git/GitHub/GOSS-GridAPPS-D-PYTHON/gov_pnnl_goss/gridlab/climate/"
    #
    # Get the current timestamp (seconds since the epoch)
    current_time = int(time.time())
    if filename:
        parser = CCodeParser(path + filename)
        parser.load_file()
        parser.parse()
        print("CLASSES-------------")
        parser.dump_classes()
        print("FUNCTIONS-----------")
        parser.dump_functions()
        print("\n")
        parse_and_convert(parser, path, filename, current_time)
        print(f"converted {len(parser.classes)}: classes and {len(parser.functions)}: functions")
        print("Done")

        # filename = "network.cpp"  # Replace with your C code file
    else:
        parser = CCodeParser()
        for filename in os.listdir(path):
            if filename.endswith(".cpp"):
                file_path = os.path.join(path, filename)
                if os.path.isfile(file_path):
                    print(f"#################################################\nOpening {filename} for conversion")
                    file_size = os.path.getsize(file_path)
                    parser.load_file(path + filename)
                    # remove all imports here
                    REMOVE_IMPORTS = True
                    if REMOVE_IMPORTS:
                        clean_code = []
                        for line in parser.c_code.split('\n'):
                            if not line.find('#include') == 0:
                                clean_code.append(line)
                    else:
                        clean_code = parser.c_code.split('\n')
                    # create a blob of code
                    code_string = '\n'.join(clean_code)
                    # remove comments
                    if file_size > GptCodeConverter.MAX_TOKENS:
                        code_string = remove_single_line_comments(code_string)
                        code_string = remove_multiline_comments(code_string)
                    print(f"File: {filename}, Orig size: {file_size}, cleaned size: {len(code_string)} (bytes)")
                    # URL-encode the text
                    # try:
                    #     code_string.encode('ascii')
                    # except UnicodeDecodeError:
                    #     raise ValueError('code is not ASCII')
                    parser.c_code = code_string
                    parser.blanked_code = code_string
                    parser.parse()
                    print("\n")
                    print("CLASSES-------------")
                    parser.dump_classes()
                    print("\n")
                    print("FUNCTIONS-----------")
                    parser.dump_functions()
                    parse_and_convert(parser, path, filename, current_time)
                    print(f"converted {len(parser.classes)}: classes and {len(parser.functions)}: functions")
                    parser.un_load_file()
        print("Done")

if __name__ == "__main__":
    directory_path = f"{os.path.expanduser('~')}/Documents/Git/GitHub/GOSS-GridAPPS-D-PYTHON/gov_pnnl_goss/gridlab/gldcore/"
    main(directory_path)
