from langchain.chat_models import ChatOpenAI
import copy
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import os
import json
import glob
import time
from dotenv import load_dotenv
from global_logger import Log
from utils.expression_tree_sitter_imp import removeMathCalls
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# chat mode instance

logger = Log.get_logger()

from utils.tree_sitter_utils import remove_comments_and_docstrings, split_functions
import os
import tiktoken


def num_tokens_from_messages(value, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        num_tokens = 0
        num_tokens += len(encoding.encode(value))
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")

def load_projects(project_folder, checkfile):
    res = {}
    filenames = [ l.strip() for l in open(checkfile, "r").readlines() ]
    print(filenames)
    for fpath in glob.glob(f"{project_folder}/**/*.sol", recursive=True):
        fname = os.path.basename(fpath)
        if fname in filenames:
            code = open(fpath).read()
            res[fname] = split_functions(code, type="dic") #code
    return res
    
def load_euqations(equations_file):
    code = open(equations_file).read()
    checkedeqs = split_functions(code, type="dict")
    res = {}
    for fn in checkedeqs:
        print(fn)
        code=" ".join(checkedeqs[fn].split("{", maxsplit=1)[1:]).strip()
        code=code[:-1]
        res[fn] = code

    return res


def create_prompt_question(user_input, file_code):
    #print(one_shot_exmaple)
    system_prompt = """You are an AI trained to detect similar code expressions. Given a Smart Contract code and a specific target code expression, your task is to find and list the most similar expressions within the provided Smart Contract code. I will show you the answer format and then please analyze the new input following code file and search for expressions that closely resemble the target code piece provided.

```
{"Answer":"Yes" or "No", "similar_expressions": [
    {
      "function_name": the matched funciton name,
      "line_number": line_number,
      "expression": the similar code
    }
  ]
  "Reason": your reason
  }



Input Smart Contract Code:
```Solidity
"""+ file_code +"""
``` 

Input Specific Target Code Expression:
```Target Expression
"""+user_input+"""
```
Please identify the similar expressions, their corresponding function name and their corresponding line numbers in the code file. You also need to replace the function calls "add", "sub", "div", "mul", "divCeil" in the found similar expressions with "+", "-", "/" and "*".  Put your results in JSON format at the beginning."""
    return system_prompt



import json
def chat_gpt_answer(checkedequations, project_codes, eq_fncs, output_dir ):
                 #checkedequations, pcode, eq_fncs, output_folder
        output_dir_path = f"{output_dir}/"
        for eq_name in eq_fncs:
            eq_code = checkedequations[eq_name]
            eq_checked_fns = []
            for (sol_file, function_name) in eq_fncs[eq_name]:
                fn_list = project_codes[sol_file]
                if function_name not in fn_list:
                    print(f"no this {function_name} , {eq_name}")
                    continue
                function_name_code = fn_list[function_name]
                eq_checked_fns.append(function_name_code)
            question = create_prompt_question(eq_code, "\n".join(eq_checked_fns))
            outputfile_question = os.path.join(f"{output_dir_path}/{eq_name}/question_end.md")
            with open(outputfile_question, "w") as f1:
                    f1.write( question )

import collections
def loadAnswers(folder):
    eq_fncs = collections.defaultdict(list)
    for eq_name in os.listdir(folder):
        for sol_file in os.listdir( os.path.join(folder, eq_name) ):
            for answer_file in os.listdir(os.path.join(folder, eq_name,sol_file ) ):
                if answer_file.startswith("answer_"):
                    print(os.path.join(folder, eq_name,sol_file, answer_file))
                    data = json.load( open( os.path.join(folder, eq_name,sol_file, answer_file), "r" ) )
                    if data["Answer"] == "Yes":
                        for m in data["similar_expressions"]:
                            function_name = m["function_name"]
                            eq_fncs[eq_name].append( (sol_file, function_name) )
    return eq_fncs

def process_solidity_chat(project_folder, euqations_file, checkfile, outfolder):
    pcode = load_projects( project_folder, checkfile )
    checkedequations = load_euqations(euqations_file)
    output_folder = os.path.join(outfolder, f"chatgpt/")
    eq_fncs = loadAnswers(output_folder)
    chat_gpt_answer(checkedequations, pcode, eq_fncs, output_folder)



import argparse
debug = False
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project", type=str, required=True, help="Project Folders")
    parser.add_argument("-e", "--equations", type=str, required=True, help="Equations")
    parser.add_argument("-c", "--checklist", type=str, required=True, help="checklist")
    parser.add_argument("-o", "--output", type=str, required=True, help="outputfolder")
    args = parser.parse_args()
    args = parser.parse_args()
    process_solidity_chat(args.project, args.equations, args.checklist, args.output)
   









    


