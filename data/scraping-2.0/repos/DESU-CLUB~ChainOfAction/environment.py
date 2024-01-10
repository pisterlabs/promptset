
import openai
import os
import subprocess
import ast
from chainofaction.agents.skillcreator import Agent
import vector_database.vector_database as skills
from chainofaction.agents.zeroShotAgent import ZeroShotAgent
import json
import pandas as pd
import random
import json
import ast
import re
from typing import Optional, List, Tuple
from math import inf
import collections
from collections import Counter
from bisect import bisect_left
import shutil
#This is just some sample code to brainstorm for the environment
random.seed = 1
def find_topmost_functions_after_class(node, class_name):
    topmost_functions = []
    
    # Recursive inner function to traverse the AST
    def traverse(node, inside_class=False):
        # Check for ClassDef and match the desired class name
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    topmost_functions.append(item.name)
            return  # Do not go further down this branch

        # Continue walking through other branches of the AST
        for child in ast.iter_child_nodes(node):
            traverse(child, inside_class)

    traverse(node)
    return topmost_functions



def extract_code_block(text):
    # Regular expression pattern to find the code block, including the language identifier
    try: 
        exec(text)
        return text
    except:
        pattern = r"```python\n(.*?)```"
        
        # Use re.DOTALL to match across multiple lines
        match = re.search(pattern, text, re.DOTALL)
        
        # Extract the code block if the pattern is found
        return match.group(1).strip() if match else None
    
def  get_function_parameters(code, function_name):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            # Found the function, now get its arguments
            args = node.args
            # Count the number of positional arguments
            num_args = len(args.args)
            # Check for *args
            vararg = 1 if args.vararg else 0
            # Check for **kwargs
            kwarg = 1 if args.kwarg else 0
            # Return the total number of arguments
            #print("Breakdown:",num_args,vararg,kwarg)
            return num_args + vararg + kwarg -1
    
    # If the function is not found, return None
    return None


    


optionalAPI = '''
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class TreeNode:
     def __init__(self, x):
         self.val = x
         self.left = None
         self.right = None
'''


def load_init_skills(path):
    passages = pd.read_csv(os.path.join(path, "leetcode.tsv"), sep='\t', header=0)
    return passages.head(int(len(passages)*5/100))

def load_dataset(path):
        passages = pd.read_csv(os.path.join(path, "leetcode.tsv"), sep='\t', header=0)
        return passages


def load_cases(path,prob_path):
    with open(os.path.join(path,prob_path)) as f:
        cases = json.load(f)
    return cases





def func_head(code):
    with open(f"chainofaction/data/code/{code[:-4]}.py","r") as f:
        coded = f.read().strip()

    pattern = r"def [^\n]*\n"
    matches = re.findall(pattern, coded)
    if matches:
        return "class Solution:"+"\n\t".join(list(map(lambda x: x.strip(),matches)))
    


class Environment:
    def __init__(self,status = None):
        if status == None:
            self.run = 0
            self.check_run()

            self.init_db()
        else:
            self.resume_run()
        self.agent = Agent(self.db, self)
        #self.agent = ZeroShotAgent(self.db, self) #Zero shot
        self.dataset = load_dataset("chainofaction/data")

    
    def check_run(self):
        #Check current directory for max run_ folders, then create a run_x+1 folder and sets self.run to x+1
        for i in os.listdir("chainofaction/data/"):
            if i.startswith("run_"):
                self.run = max(self.run,int(i[4:]))
        self.run+=1
        os.mkdir(f"chainofaction/data/run_{self.run}")
        os.mkdir(f"chainofaction/data/run_{self.run}/newdesc")
        os.mkdir(f"chainofaction/data/run_{self.run}/tracker")
        os.mkdir(f"chainofaction/data/run_{self.run}/vis")
    def reset(self):
        self.db = self.init_db()
        self.Agent = Agent(self.db)

    def init_db(self):
        dataset = "leetcode"
        emb_func = skills.MiniLML6V2EmbeddingFunction()
        data_dir = "chainofaction/data"
        docs = load_init_skills(data_dir)
        self.running_id = len(docs)
        for i in docs["title"]:
            shutil.copy(f"chainofaction/data/code/{i[:-4]}.py",f"chainofaction/data/run_{self.run}/{i[:-4]}.py")
        docs["indextext"] = docs["title"].astype(str) + "\n" + docs["problem_text"] + "\n" + docs["skill_description"]
        self.db= skills.ChromaWithUpsert(
        name=f"{dataset}_minilm6v2",
        embedding_function=emb_func,  # you can have something here using /embed endpoint
        persist_directory= "chainofaction/data/"
        )
        if self.db.is_empty():
            self.db.upsert_texts(
                texts=docs.indextext.tolist(),
                # we handle tokenization, embedding, and indexing automatically. You can skip that and add your own embeddings as well
                metadata=[{'id': id, 'title': title, 'problem_text': problem_text, 'skill_description': skill_description}
                        for (id, title, problem_text, skill_description) in
                        zip(docs.id, docs.title, docs.problem_text, docs.skill_description)],  # filter on these!
                ids=[str(i) for i in docs.id],  # unique for each doc
            )


    def step(self,problem, title):
        
        if problem == None:
            return None
        with open(f"chainofaction/data/run_{self.run}/tracker/{title[:-4]}.txt",'w') as f:
            f.write("created")
        cases = load_cases("chainofaction/data/fullcases",title[:-3]+'json')
        fn_head = func_head(title)
        soln = self.agent.get_response(problem,cases, fn_head,title)
        header = func_head(title)
        if soln != None:
            code, desc, title = soln
            if code != None:
                with open(f"chainofaction/data/run_{self.run}/{title[:-4]}.py",'w') as f:
                    f.write(code)
                texts = "\n".join([str(self.running_id),str(title),(problem),(desc)])
                #print(self.running_id, title, problem, desc)
                self.db.upsert_texts(texts, 
                                    metadata = [{"id": self.running_id, "title":title,"problem_text":problem,"skill_description":desc}\
                                                ], ids = [str(self.running_id)])
                self.running_id+=1
        return soln

    #This is the main function
    #What I plan to do is:
    #1. Retrieve a sample from the dataset
    #2. Check if sample qn requires input
    #3. How do I check for this (('m'))
    #4. If needs, will loop through all the test cases

   
    ##Reuse code from test.py
    def execute(self, code, cases):
        #placeholder
        for i in cases:
            try:
                code = extract_code_block(code)
                #print(code)
                if code ==None:
                    return ("No code found", False)
                exec(code, globals())               
                parsed_tree = ast.parse(code)
                fn_head = find_topmost_functions_after_class(parsed_tree, "Solution")[0]
                if fn_head != None:
                    #print((f'Solution().{fn_head}({i["input"]})'))
                    result = eval((f'Solution().{fn_head}(*i["input"])'))
                    
                else:
                    result = ("Code is not encapsulated in a function", False)
                    return result
                #print(result, i["output"],type(result),type(i["output"]))
                if str(result) == str(i["output"]):
                    continue
                else:
                    return (f"Failed for input: {i['input']}\n Expected output: {i['output']}\nCurrent output: {result}",False)
            except Exception as e:
                raise e
                return (f"Error in code: {e}", False)
            
        return (None, True)
    

    def main(self):
        tasks = self.dataset['title'].tolist()
        random.shuffle(tasks)
        for title in tasks:
            if title not in os.listdir(f"chainofaction/data/run_{self.run}") and title not in os.listdir(f"chainofaction/data/run_{self.run}/tracker"):
                problem = self.dataset[self.dataset['title']==title]['problem_text'].tolist()[0]
                print(problem) 
                self.step(problem,title)
                    
    def resume_run(self):
        self.run = 0
        for i in os.listdir("chainofaction/data/"):
            if i.startswith("run_") and i[4:].isdigit():
                self.run = max(self.run,int(i[4:]))
        self.resume_db()

    def resume_dataset(self):
            desc = os.listdir(f"chainofaction/data/run_{self.run}/newdesc")
            titles = list(filter(lambda x: x.endswith(".py"),os.listdir(f"chainofaction/data/run_{self.run}")))
            titles = list(map(lambda x: x[:-3]+".txt",titles))
            titles = list(set(titles).intersection(set(desc)))
            problems = list(set(os.listdir(f"chainofaction/data/problems")).intersection(set(titles)))
            for i in range(len(problems)):
                with open(f"chainofaction/data/problems/{problems[i]}") as f:
                    problem = f.read()
                problems[i] = problem
            print(len(problems),len(titles),len(desc))
            return pd.DataFrame({"id":list(range(len(problems))),"title":titles,"problem_text":problems,"skill_description":desc})
    def resume_db(self):
        dataset = "leetcode"
        emb_func = skills.MiniLML6V2EmbeddingFunction()
        fullLst = []
        for i in os.listdir(f"chainofaction/data/run_{self.run}"):
            if i.endswith(".py"):
                fullLst.append(i[:-3]+"txt")
        docs = self.resume_dataset()
        self.running_id = len(docs)+1
        docs["indextext"] = docs["title"].astype(str) + "\n" + docs["problem_text"] + "\n" + docs["skill_description"]
        self.db= skills.ChromaWithUpsert(
        name=f"{dataset}_minilm6v2",
        embedding_function=emb_func,  # you can have something here using /embed endpoint
        persist_directory= "chainofaction/data/"
        )
        if self.db.is_empty():
            self.db.upsert_texts(
                texts=docs.indextext.tolist(),
                # we handle tokenization, embedding, and indexing automatically. You can skip that and add your own embeddings as well
                metadata=[{'id': id, 'title': title, 'problem_text': problem_text, 'skill_description': skill_description}
                        for (id, title, problem_text, skill_description) in
                        zip(docs.id, docs.title, docs.problem_text, docs.skill_description)],  # filter on these!
                ids=[str(i) for i in docs.id],  # unique for each doc
            )
        
        
                


""" data = load_data("chainofaction/data")
prob = random.choice(data['title'].tolist())
print(prob)
cases = load_cases("chainofaction/data/fullcases",prob)
for case in cases:
    print(case)
    act = eval(case)
    print(act[1])
    func_inp = act[0]
    func_out = act[1]
    print(type(func_out))
 """

env = Environment("res")

env.main()