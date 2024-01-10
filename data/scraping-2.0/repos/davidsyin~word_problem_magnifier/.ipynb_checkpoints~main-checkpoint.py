import openai
import json
import os
import string
import random


def clear():
    os.system('clear')


def to_prompt(question):
    return "For the question " + question + " , multiply all numbers in the question by 2, then tell me the modified question and the final number in its answer. Do not say anything else."


class ChatBot:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages = []
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        # print(self.messages)
        return result

    def execute(self):
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=self.messages)
        # Uncomment this to print out token usage each time, e.g.
        # {"completion_tokens": 86, "prompt_tokens": 26, "total_tokens": 112}
        # print(completion.usage)
        return completion.choices[0].message.content


openai.api_key = "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"
verifier = ChatBot()
errors = 0
banned_vars = {'this', 'time', 'line_idx', 'start_idx', 'end_idx', 'random', 'words', 'is_loop', 'multiple_reses',
               'output_lines', 'curr_output', 'output_line', 'j', 'string', 'banned_vars', 'In', 'Out', 'get_ipython',
               'exit', 'quit', 'open', 'openai', 'json', 'os', 'clear', 'to_prompt', 'ChatBot', 'verifier', 'errors',
               'sys', 'io', 'debug', 'indent', 'i', 'statement', 'lines', 'curr_code', 'noprint_code', 'vars',
               'var_map', 'vars_list', 'vars_it', 'skip', 'res', 'line', 'var_found', 'it', 'variables', 'variable',
               'var', 'old_stdout', 'buffer', 'whatWasPrinted', 'print_res', 'new_code', 'ans', 'output', 'outfile',
               'potential_var', 'found'}
import sys
import io

words = ["cupboard", "dog", "cat", "funny", "pillow", "coffee maker", "bed", "spoon", "blanket", "knife", "stove",
         "sink", "washing machine", "pot", "dish", "fridge", "sofa", "stool", "cup", "fork", "glass", "attractive",
         "bald", "beautiful", "chubby", "clean", "dazzling", "drab", "elegant", "fancy",
         "fit", "flabby", "glamorous", "gorgeous", "handsome", "long", "magnificent", "muscular", "plain", "plump",
         "quaint", "scruffy", "shapely", "short", "skinny", "stocky", "ugly", "unkempt", "unsightly", "aggressive",
         "agreeable", "ambitious", "brave", "calm", "delightful", "eager", "faithful", "gentle",
         "happy", "jolly", "kind", "lively", "nice", "obedient", "polite", "proud", "silly", "thankful", "victorious",
         "witty", "wonderful", "zealous", "angry", "bewildered", "clumsy", "defeated", "embarrassed", "fierce",
         "grumpy", "helpless", "itchy", "jealous", "lazy", "mysterious", "nervous", "obnoxious",
         "panicky", "pitiful", "repulsive", "scary", "thoughtless", "uptight", "worried", "broad", "chubby", "crooked",
         "curved", "deep", "flat", "high", "hollow", "low", "narrow", "refined", "round", "shallow", "skinny", "square",
         "steep", "straight", "wide", "big", "colossal", "fat", "gigantic", "great", "huge",
         "immense", "large", "little", "mammoth", "massive", "microscopic", "miniature", "petite", "puny", "scrawny",
         "short", "small", "tall", "teeny", "tiny"]
debug = []
indent = False
is_loop = False
for i in range(1):
    # statement=verifier("Generate a "+random.choice(words)+" non-function, non-class Python code. ")
    statement = '''joke = \"Why don't scientists trust atoms?\\nBecause they make up everything!\"\nprint(joke)\n\n\n'''
    start_idx = 0
    end_idx = len(statement) - 1
    for j in range(len(statement)):
        if j < len(statement) - 3 and (statement[j:j + 3] == "```" or statement[j:j + 3] == "'''"):
            if start_idx:
                end_idx = j
                break
            if j < len(statement) - 9 and statement[j + 3:j + 9] == "python":
                start_idx = j + 9
            else:
                start_idx = j + 3
    statement = statement[start_idx:end_idx + 1]
    statement = statement.replace("    ", "")
    statement = statement.replace("`", "")
    if "def" in statement or "input" in statement or "random" in statement or "turtle" in statement:
        continue
    lines = statement.split("\n")
    lines.append("\n")
    curr_code = ""
    noprint_code = ""
    vars = {"temp"}
    var_map = {"temp": ""}
    vars_list = []
    vars_it = {"temp"}
    skip = False
    res = {"question": lines, "answer": [], "stdout": []}
    line_idx = 0
    for line in lines:
        var_found = False
        if len(line) == 0 or line[0] == "#":
            continue
        if len(line.split()) > 1 and line.split()[0] in ["for", "if", "while"]:
            potential_var = line.split()[1]
            if potential_var not in vars and not var_found:
                var_found = True
                vars_list.append(potential_var)
                vars_it.add(potential_var)
                vars.add(potential_var)
        if indent:
            found = False
            for var in vars_it:
                for j in range(len(line) - len(var)):
                    if line[j:j + len(var)] == var and (j == len(line) - len(var) - 1 or (
                            line[j + len(var)] not in string.ascii_letters and line[j + len(var)] != "_")):
                        found = True
                        break
                if found:
                    break
            indent = found or "for" in lines[line_idx - 1] or "if" in lines[line_idx - 1] or "while" in lines[
                line_idx - 1] or "else" in lines[line_idx - 1]
        it = 0
        if indent and it == 0:
            curr_code += "    "
            noprint_code += "    "
        curr_code += line + "\n"
        if "print" not in line:
            noprint_code += line + "\n"
        else:
            noprint_code += "pass" + "\n"
        try:
            exec(noprint_code)
        except:
            debug = []
        res["question"] = curr_code
        variables = list(locals())
        for variable in variables:
            if variable[0] != "_" and variable not in vars and variable not in banned_vars and variable in statement.split(" "):
                vars.add(variable)
                vars_list.append(variable)
            it += 1
        if len(vars_list):
            try:
                ans = " [ "
                for var in vars_list:
                    if not is_loop and var in vars_it:
                        continue
                    old_stdout = sys.stdout  # Memorize the default stdout stream
                    sys.stdout = buffer = io.StringIO()
                    new_code = noprint_code + "\n"
                    if indent:
                        new_code += "    "
                    new_code += "print(" + var + ")\n"
                    new_code += '''print("SEP")'''
                    exec(new_code)
                    sys.stdout = old_stdout  # Put the old stream back in place
                    whatWasPrinted = buffer.getvalue()  # Return a str containing the entire contents of the buffer.
                    clear()
                    print_res = True
                    if var in var_map and var_map[var] == whatWasPrinted:
                        print_res = False
                    output_lines = whatWasPrinted.split("SEP")
                    if output_lines[-1] == "":
                        output_lines = output_lines[:-1]
                    if print_res:
                        if len(output_lines) == 1:
                            if not (var in var_map and var_map[var].replace("\n", "") == output_lines[-1]):
                                ans += var + " : " + whatWasPrinted
                        else:
                            ans += " ( "
                            curr_output = ""
                            found = False
                            for output_line in output_lines:
                                if output_line == curr_output or len(output_line) == 0 or output_line == "\n":
                                    continue
                                if len(output_line.replace(" ", "")) == 0:
                                    ans += var + " : " + "None "
                                    curr_output = output_line
                                else:
                                    ans += var + " : " + output_line + " "
                                    curr_output = output_line
                            ans = ans[:-1]
                            ans += " ) "
                            if ans == ' [  ( ) ':
                                ans = ""
                    var_map[var] = output_lines[-1]
                ans += " ] "
                if ans == " [  ] ":
                    ans = "None"
                res["answer"].append(ans)
            except:
                res["answer"].append("Error")
        else:
            res["answer"].append("None")
        try:
            res["stdout"] = []
            old_stdout = sys.stdout  # Memorize the default stdout stream
            sys.stdout = buffer = io.StringIO()
            exec(curr_code)
            sys.stdout = old_stdout  # Put the old stream back in place
            whatWasPrinted = buffer.getvalue()  # Return a str containing the entire contents of the buffer.
            clear()
            multiple_reses = len(whatWasPrinted.split("\n")) > 1
            ans = ""
            if multiple_reses:
                ans += " ( "
            for output in whatWasPrinted.split("\n"):
                if len(output) == 0:
                    ans += "None\n"
                    continue
                ans += output + "\n"
            if multiple_reses:
                ans += " ) "
            res["stdout"].append(ans)
        except:
            res["stdout"].append("Error")
        if line.split(" ")[0] in ["for", "if", "while", "elif", "else"]:
            indent = True
        if line.split(" ")[0] in ["for", "if"]:
            is_loop = True
        line_idx += 1
    # with open("/home/mcwave/data/python/code-v2.json", "a") as outfile:
    #     json.dump(res, outfile)
    #     outfile.write("\n")
    for var in vars_list:
        new_code = "del " + var
        exec(new_code)
debug = []
