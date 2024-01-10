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
            res[fname] = code
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

def creat_questions(euqation, file_code):
    file_code = remove_comments_and_docstrings(file_code)
    question_list = [ (create_prompt_question(euqation, file_code), file_code) ]
    def split_list(input_list, chunk_size=15):
        return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]

    if num_tokens_from_messages(question_list[0][0]) > 4097:
        code_fn = split_functions(file_code)
        codes_chunk = split_list(code_fn)
        code_pieces = []
        for chunk in codes_chunk:
            fns = [ ]
            for k in chunk:
                fns.append( k )
            code_pieces.append("\n".join(fns))
        question_list = [ (create_prompt_question(euqation, cpiece ), cpiece) for cpiece in code_pieces ]
        
    return question_list


import json
def chat_gpt_answer(data_eggs, project_codes, output_dir ):
        output_dir_path = f"{output_dir}/"
        os.makedirs(os.path.join(f"{output_dir_path}/"), exist_ok=True)
        for code_file in project_codes:
            name = os.path.basename( code_file )
            file_code = project_codes[name]
            for eq_name, ex in data_eggs.items():
                euqation = ex
                eid = eq_name
                question_list = creat_questions(euqation, file_code)
                for i, question_pair in enumerate(question_list):
                    question, code_piece = question_pair[0], question_pair[1]
                    os.makedirs(os.path.join(f"{output_dir_path}/{eid}/{name}"), exist_ok=True)
                    outputfile_answer = os.path.join(f"{output_dir_path}/{eid}/{name}/answer_{i}.md")
                    outputfile_question = os.path.join(f"{output_dir_path}/{eid}/{name}/question_{i}.md")
                    outputfile_info = os.path.join(f"{output_dir_path}/{eid}/{name}/info.json")
                    outputfile_code_piece = os.path.join(f"{output_dir_path}/{eid}/{name}/code_piece_{i}.sol")
                    messages = [
                        SystemMessage(content=question),
                        ]
                    max_tokens = 4097 - num_tokens_from_messages(question)-150
                    chat = ChatOpenAI(temperature=0, max_tokens=max_tokens)
                    try:
                        if debug:
                            response = "debug message" #chat(messages)
                            logger.info(response)
                            answers = "debug answers" #response.content
                        else:
                            response = chat(messages)
                            logger.info(response)
                            answers = response.content
                        with open(outputfile_answer, "w") as f:
                            f.write( answers )
                        with open(outputfile_question, "w") as f1:
                            f1.write( question )
                        with open(outputfile_code_piece, "w") as f2:
                            f2.write( code_piece )
                        info = {"eq": euqation, "eid":eid, "file": code_file }
                        json.dump(info, open(outputfile_info, "w"), indent=4)
                        if not debug:
                            time.sleep(3)
                    except Exception as e:
                        logger.info(e)
                        continue


def process_solidity_chat(project_folder, euqations_file, checkfile, outfolder):
    pcode = load_projects( project_folder, checkfile )
    checkedequations = load_euqations(euqations_file)
    output_folder = os.path.join(outfolder, f"chatgpt/")
    chat_gpt_answer(checkedequations, pcode, output_folder)
    
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
   









    


