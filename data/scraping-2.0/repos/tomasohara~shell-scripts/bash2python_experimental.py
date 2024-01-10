#!/usr/bin/env python3
#
# TODO: Refine overview comments (e.g., notes and technique used)
# Note: no need to get detailed. Just write down main intention and some ideas how it will work.
#
# This script converts from Bash snippets to Python. This is not intended as a general purpose
# conversion utility, which is a major undertaking given the complexities of Bash syntax.
# Instead, this is intended to capture commonly used constructs, such as by Tom O'Hara
# during shell interactions.
#
# Notes:
# - This just outputs the Python code, along with import for helper module and initialization.
#   The user might have to do some minor fixup's before the code will run properly.
# - Simple Bash statements get converted into shell invocation calls (a la os.system).
#     pushd ~/; cp -f .[a-z]* /tmp; popd     =>  run("pushd ~/; cp -f .[a-z]* /tmp; popd")
# - Simple variable assignments get translated directly, but complex runs are converted into echo command.
#     log="run-experiment.log";              =>  log = "run-experiment.log"
#     today=$(date '+%d%b%y')                =>  today = run("echo \"date '+%d%b%y'\"")
# - Simple loops get converted into Python loops, namely for-in and while:
#     for v in abc def: do echo $v; done     =>  for v in ["abc", "def"]: gh.run("echo {v}")
# - Unsupported or unrecognized constructs are flagged as runtime errors:
#     for (( i=0; i<10; i++ )); do  echo $i; done
#         =>
#     # not supported:  for (( i=0; i<10; i++ )); do  echo $i; done
#     raise NotImplementedError()
#
# TODO:
# - Flag constructs not yet implemented:
#   -- C-style for loops (maybe use cursed-for module)
#   -- Bash arrays and hashes #Tana-note: Working on this in a separate file
# - Add more sanity checks (e.g., via debug.assertion).
#

# TODO refine a little
"""Bash snippet to Python conversion"""

# Standard modules
from collections import defaultdict
import os
import re
import click
import openai
import time
# Local modules
import mezcla
from mezcla import debug
from mezcla import glue_helpers as gh
from mezcla.main import Main
from mezcla.my_regex import my_re
from mezcla import system
from mezcla.text_utils import version_to_number
from mezcla.glue_helpers import run_via_bash as run

# Version check
debug.assertion(version_to_number("1.3.4") <= version_to_number(mezcla.__version__))

# Environment constants
USER = system.getenv_text("USER", "unknown-user",
                          "User ID")
USE_MEZCLA = system.getenv_bool("USE_MEZCLA", (USER == "tomohara"),
                                "Whether to use mezcla support")
INIT_FILE = system.getenv_value("INIT_FILE", None,
                                "File to source before running Bash command")

# Global settings
if USE_MEZCLA:
    re = my_re

## TEMP:
## NOTE: Eventually most pylint issues should be resolved (excepting nitpicking ones)
## pylint: disable=no-self-use,unneeded-not

PYTHON_HEADER = """# Output from bash2python.py
'''Python code from Bash snippet'''
from mezcla.glue_helpers import run_via_bash
from mezcla import system

INIT_FILE = system.getenv_value("INIT_FILE", None,
                                "File to source before running Bash command")

def run(command, skip_print=False):
    '''Runs COMMAND and return output. Also, prints non-empty output unless SKIP_PRINT'''
    result = run_via_bash(command, init_file=INIT_FILE)
    if (not skip_print) and result:
       print(result)
    return result
"""


def get_bash_var_hash():
    """Return a lookup hash for checking whether Bash variable is defined
    Note: this includes environment variables as well as regular ones"""
    # Sample listing from Bash set command:
    #   ANACONDA_HOME=/
    #   BASH=/bin/bash
    #   ...
    #   zless ()
    #   {
    #       zcat "$@" | $PAGER
    #   }

    # Extract variables from set command output
    var_hash = defaultdict(bool)
    bash_variable_listing = gh.run_via_bash("set", init_file=INIT_FILE)
    for line in bash_variable_listing.splitlines():
        if my_re.search(r"^([A-Za-z0-9_]+)=", line):
            var_hash[my_re.group(1)] = True

    # Run sanity checks
    if debug.debugging():
        env_vars = sorted(list(os.environ.keys()))
        bash_vars = sorted(list(var_hash.keys()))
        ## TODO: debug.trace_expr(5, bash_vars, env_vars, max_len=2048, sep="\n")
        debug.trace_expr(5, bash_vars, max_len=4096)
        debug.trace_expr(5, env_vars, max_len=4096)
        debug.assertion(not system.difference(env_vars, bash_vars))
        debug.assertion(sorted(system.intersection(env_vars, bash_vars)) == env_vars)
    debug.trace(6, f"get_bash_var_hash() => {var_hash}")
    return var_hash


class Bash2Python:
    """Returns a Python-like file based on Bash input"""
    KEYWORD_MAP = {"function": "def"}
    LOOP_CONTROL = ["break", "continue"]

    def __init__(self, bash, shell):
        self.cmd = bash
        self.exec = shell
        self.bash_var_hash = get_bash_var_hash()
        self.variables = []

    def map_keyword(self, line):
        """Perform conversion for single keyword statement"""
        in_line = line
        if my_re.search(r"^(\s*)(\w+)(.*)$", line):
            indent = my_re.group(1)
            keyword = my_re.group(2)
            remainder = my_re.group(3)
            line = indent + self.KEYWORD_MAP.get(keyword, keyword) + remainder
        debug.trace(5, f"map_keyword({in_line!r}) => {line!r}")
        return line

    def for_in(self, line):
        if my_re.search(r"for .* in (.* )", line):
            value = my_re.group(1)
            values = value.replace(" ", ", ")
            values = f"[{values}]"
            line = line.replace(value, values)
        if my_re.search(r"\{([0-9]*)\.\.([0-9]*)\}", line):
            values = my_re.group(0)
            numberone, numbertwo = my_re.group(1, 2)
            line = line.replace(values, f"range({numberone}, {numbertwo})")
        return line

    def codex(self, line):
        """Uses OpenAI Codex to translate Bash to Python"""
        # Apply for an API Key at https://beta.openai.com
        openai.api_key = "YOUR_API_KEY"
        # Define the code generation prompt
        prompt = f"Convert this Bash snippet to a one-liner Python snippet: {line}"
        # Call the Codex API
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=3*len(line),
            n=1,
            stop=None,
            temperature=0.6, # more of this makes response more inestable
        )

        # Get the generated code
        i = 0
        while i < len(response["choices"]):
            # Check if the text of the choice matches the input
            code = response["choices"][i]["text"].strip().replace("\n", "")
            if code in line:
                time.sleep(2)
                # If a match is found, access the next choice
                if i + 1 < len(response["choices"]):
                    code = response["choices"][i + 1]["text"].strip().replace("\n", "")
                    break
            # Increment the index
            i += 1
        comment = "#" + code
        return comment

    def process_compound(self, line, cmd_iter):
        # Declare loop and statements as a tuple
        loop_count = 0
        body = line

        def recursive_loop(line, cmd_iter, loop_count, body):
            for_loop = ("for", "do", "done")
            while_loop = ("while", "do", "done")
            if_statement = ("if", "then", "fi")
            elif_statement = ("elif", "then", "fi")
            loops = (for_loop, while_loop, if_statement, elif_statement)
            converted = False
            for loop in loops:
                if my_re.search(fr"\s*\[?\s*{loop[0]} *(\S.*)\s*\]?; *{loop[1]}", line):
                    debug.trace(4, f"Processing {loop[0]} loop")
                    var = self.var_replace(my_re.group(1), is_loop=True)
                    var = self.operators(var)
                    if body != line:
                        body += "    " * (loop_count)
                        body += f"{loop[0]} {var}:\n"
                    else:
                        body = f"{loop[0]} {var}:\n"
                    loop_count += 1
                    while loop_count > 0:
                        loop_line = next(cmd_iter, "")
                        debug.trace_expr(5, loop_line)
                        if loop[0] == "if" and "else" in loop_line:
                            body += "    " * (loop_count)
                            body += "else:\n"
                        if loop_line is None or loop_line == "\n":
                            loop_line = ""
                            loop_count -= 1
                        if re.search(r'^\s*#', loop_line):
                            body += loop_line
                            continue
                        if my_re.search(fr"\s*{loop[2]}\)?", loop_line):
                            loop_line = ""
                            loop_count -= 1
                        _, body = recursive_loop(loop_line, cmd_iter, loop_count, body)
                        if _ is True:
                            continue
                        loop_line = loop_line.strip("\n").strip(";")
                        if loop_line.strip() in self.LOOP_CONTROL:
                            body += loop_line + "\n"
                            body += self.map_keyword(loop_line) + "\n"
                        elif loop_line.strip():
                            (converted, loop_line) = self.process_simple(loop_line)
                            if converted:
                                body += loop_line + "\n"
                            else:
                                body += self.var_replace(loop_line.strip("\n"),
                                                         indent="     ") + "\n"
                        converted = True
            return converted, body

        converted, line = recursive_loop(line, cmd_iter, loop_count, body)
        # debug.trace(7, f"process_compound({line!r}) => ({converted}, {line!r})")
        return (converted, line)

    def var_replace(line, other_vars=None, indent=None, is_condition=False, is_loop=False):
        """Replaces bash variables with python variables and also derive run call for LINE
        Notes:
        - Optional converts OTHER_VARS in line.
        - Use INDENT to overide the line space indentation.
        """
        # Initialize variables
        comment = ""
        if indent is None:
            indent = ""
        # Convert all quotes
        line = line.replace("'", '"')

        # Replace bash variables with python variables
        def variable(line):
            bash_vars = re.findall(r'\$\w+', line)  # finds bash variables
            if other_vars:
                bash_vars += other_vars
            for var in bash_vars:
                if (var[1:] in self.bash_var_hash) and (var not in self.variables):
                    debug.trace(4, "Excluding Bash-defined variable {var}")
                else:
                    line = line.replace(var, "{" + var[1:] + "}")
            return "f'" + line + "'" if bash_vars else line

        def command(line):
            # Replace bash commands with python commands
            has_command = False
            bash_commands = re.findall(r'\$\((.*?)\)', line)  # finds bash commands
            if bash_commands:
                line = "echo " + line
                line = line + comment
                has_command = True
            return line, has_command

        def defaults(line):
            # Replace default values
            has_default = False
            defaults = re.findall(r'\$\{\w+:-[^\}]+\}', line)  # finds all bash variables with default values
            if defaults:
                var_name = re.search(r'\$\{(\w+):-[^\}]+\}', line).group(1)
                var_default = re.search(r'\$\{\w+:-(.*)\}', line).group(1)
                line = line.replace(line, f"{{{var_name} if {var_name} is not None else '{var_default}'}}")
                has_default = True
            return line, has_default

        # Derive indentation
        if my_re.search(r"^(\s+)(.*)", line):
            if not indent:
                indent = my_re.group(1)
            line = my_re.group(2)

        # Check for assignment and convert to list
        assignment = my_re.search(r"(\S)*(\s)*=(\s)*(\S)*", line)
        if assignment and not is_condition:  # if the line is a variable declaration
            var = [var for var in assignment.group(1).split() if "$" in var]
            self.variables.extend(var)
        line = line.split('=')

        for i in range(len(line)):
            check_line = line[i]
            line[i], has_default = defaults(line[i])
            if not has_default:
                line[i], has_command = command(line[i])
                line[i] = variable(line[i])
                if has_command:
                    line[i] = "run('" + line[i] + "')"
            has_nothing = (check_line == line[i])
            if has_nothing and not assignment:
                line[i] = "run('" + line[i] + "')"
        # Reunify line if assignment
        if assignment:
            line = ' = '.join(line)
        else:
            line = ''.join(line)
        line = indent + line
        # debug.trace(5, f"var_replace({in_line!r}, othvar={other_vars} ind={indent} cond={is_condition}) => {line!r}")
        return line

    def operators(self, line):
        """Returns line with operators converted to Python equivalents"""
        # Dictionary with Bash operators and their Python equivalents
        operators = {"=": " == ",
                     "!=": " != ",
                     "!": "not ",
                     "-eq": " == ",
                     "-e": " os.path.exists ",
                     "-ne": " != ",
                     "-gt": " > ",
                     "-ge": " >= ",
                     "-lt": " < ",
                     "-le": " <= ",
                     "-z": " '' == ",
                     "-n": " '' != ",
                     "&&": " and ",
                     "\|\|": " or ",  # NOTE: need to escape | for Python
                     }

        in_line = line
        # Iterate over operators and replace them with Python equivalents
        for bash_operator, python_equivalent in operators.items():
            line = re.sub(rf"(\S*) *{bash_operator} *(\S*)", fr"\1{python_equivalent}\2", line).replace("[", "").replace("]", "")
        # Replace Bash true/false statements with Python equivalent
        line = re.sub("\[ 1 \]", "True", line)
        line = re.sub("\[ 0 \]", "False", line)
        debug.trace(5, f"operators({in_line!r}) => {line!r}")
        return line

    def process_simple(self, line):
        """Process simple statement conversion for LINE"""
        debug.trace(6, f"in process_simple({line!r})")
        in_line = line
        converted = False
        # Convert miscellenous commands
        # - break
        # TODO: continue (dont think this is needed)
        debug.trace(6, "checking miscellenous")
        if (line.strip() == "break"):
            debug.trace(4, "processing break")
            converted = True
        # - arithmetic expression
        #   (( expr ))
        if my_re.search(r"^(\s*)\(\( (.*) \)\)\s*$", line):
            debug.trace(4, "processing arithmetic expression")
            indent = my_re.group(1)
            expression = my_re.group(2)
            line = (indent + expression)
            converted = True
        if re.search(r"let \"(\S*)\"", line):
            debug.trace(4, "processing let")
            line = re.sub(r"let \"(\S*)\"", r"\1", line)
            converted = True
        debug.trace(7, f"process_simple({in_line!r}) => ({converted}, {line!r})")
        return (converted, line)

    def format(self):
        """Convert self.cmd into python, returning text"""  # TODO: refine
        # Tom-Note: This will need to be restructured. I preserved original for sake of diff.
        python_commands = []
        cmd_iter = (system.open_file(self.cmd) if system.file_exists(self.cmd)
                    else iter(self.cmd.splitlines(keepends=True)))
        if cmd_iter:
            for line in cmd_iter:  # for each line in the script
                debug.trace_expr(5, line)
                if (not line.strip()) or re.search(r"^\s*#", line) or line.startswith("{") or line.startswith("}"):
                    python_commands.append(line.strip("\n"))
                    continue
                line = line.strip("\n")
                # if comment line, skip
                comment = ""
                if "#" in line:
                    line, comment = line.split("#", 1)
                    if not line:
                        return "\n".join(python_commands)
                line = line[:-1] if ";" == line[-1] else line  # remove the ";" last character
                (converted, line) = self.process_compound(line, cmd_iter)
                if not converted:
                    line = self.var_replace(line)
                    (converted, line) = self.process_simple(line)
                # Adhoc fixups
                if comment:
                    line += f" # {comment}"
                python_commands.append(line)
        return "\n".join(python_commands)

    def header(self):
        """Returns Python header to use for converted snippet code"""
        return PYTHON_HEADER


# -------------------------------------------------------------------------------
@click.command()
@click.option("--script", "-s", help="Script or snippet to convert")
@click.option("--output", "-o", help="Output file")
@click.option("--overview", help="List of what is working for now")
@click.option("--execute", is_flag=True, help="Try to run the code directly (probably brokes somewhere)")
@click.option("--line-numbers", is_flag=True, help="Add line numbers to the output")
def main(script, output, overview, execute, line_numbers):
    """Entry point"""
    if overview:
        print("""Working: 
            -If, elif, else, and while.
            -Variable Assignments, all of them. 
            -Piping to file. (Uses bash for it)
            -All kind of calls to system. 
            -Printing (still using run(echo) for it)
            -AI sugestions using OpenAI GPT-3 Codex (requires API key)
            -Bash defaults
Not working yet: 
            -For loops (there is an untested function)")
            -Writing on / reading files
            -Bash functions
            -Any kind of subprocess
            -C-style loops """)
    debug.trace(3, f"main(): script={system.real_path(__file__)}")
    bash_snippet = script
    if not bash_snippet:
        print("No script or snippet specified. Use --help for usage or --script to specify a script or snippet")
        return
    if line_numbers:
        with open(script, 'r') as infile, open(script + ".b2py", 'w') as outfile:
            # Loop through each line in the input file
            for i, line in enumerate(infile):
                # Write the modified line to the output file
                if line.startswith("#"):
                    outfile.write(line)
                elif line == "\n":
                    outfile.write(line)
                else:
                    outfile.write(f"{line}#b2py #Line {i}" + "\n")
        bash_snippet = script + ".b2py"

    # Show simple usage if --help given
    if USE_MEZCLA:
        dummy_main_app = Main(description=__doc__, skip_input=False, manual_input=False)
        debug.assertion(dummy_main_app.parsed_args)
        bash_snippet = dummy_main_app.read_entire_input()

    # Convert and print snippet
    b2p = Bash2Python(bash_snippet, "bash -c")
    if output:
        with open(output, "w") as out_file:
            out_file.write(b2p.header())
            out_file.write(b2p.format())
    if execute:
        cmd = b2p.format()
        print(f"# {cmd}")
        print(eval(str(cmd)))
    else:
        print(b2p.header())
        print(b2p.format())


# -------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
