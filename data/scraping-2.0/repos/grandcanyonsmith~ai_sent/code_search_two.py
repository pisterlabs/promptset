from utils import *
import os
from glob import glob
import pandas as pd
import openai
openai.api_key = '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'

def get_function_name(code):
    """
    Extract function name from a line beginning with "def "
    """
    assert code.startswith("def ")
    return code[len("def "): code.index("(")]

def get_until_no_space(all_lines, i) -> str:
    """
    Get all lines until a line outside the function definition is found.
    """
    ret = [all_lines[i]]
    for j in range(i + 1, i + 10000):
        if j < len(all_lines):
            if len(all_lines[j]) == 0 or all_lines[j][0] in [" ", "\t", ")"]:
                ret.append(all_lines[j])
            else:
                break
    return "\n".join(ret)

def get_functions(filepath):
    """
    Get all functions in a Python file.
    """
    whole_code = open(filepath).read().replace("\r", "\n")
    all_lines = whole_code.split("\n")
    for i, l in enumerate(all_lines):
        if l.startswith("def "):
            code = get_until_no_space(all_lines, i)
            function_name = get_function_name(code)
            yield {"code": code, "function_name": function_name, "filepath": filepath}

# get user root directory
path = input("Enter the path to the directory you want to search: ")
search_d = path

# REMOVE THE FILE FROM THE PATH
def remove_file_from_path(path):
    return path[:path.rfind('/')]

folder = remove_file_from_path(path)
print(folder)
root_dir = "/Users/canyonsmith/Desktop/sentient_ai/assistent_ai_code/whispering/"
search_dir = root_dir + search_d
root_dir = "/Users/canyonsmith/Desktop/sentient_ai/assistent_ai_code/whispering/" + search_d






print("Enter the path to the file you want to search: ")
print(root_dir)


# path to code repository directory
code_root = root_dir
print(code_root)
code_files = [y for x in os.walk(code_root) for y in glob(os.path.join(x[0], '*.py'))]
# only look at files in serverless folder, stipulations folder, and utils folder
# code_files = [y for x in os.walk(code_root) for y in glob(os.path.join(x[0], '*.py')) if any([i in x[0] for i in ['serverless', 'stipulations', 'utils']])]

print("Total number of py files:", len(code_files))
all_funcs = []
for code_file in code_files:
    funcs = list(get_functions(code_file))
    all_funcs.extend(iter(funcs))
print("Total number of functions extracted:", len(all_funcs))

from openai.embeddings_utils import get_embedding

df = pd.DataFrame(all_funcs)
df['code_embedding'] = df['code'].apply(lambda x: get_embedding(x, engine='code-search-babbage-code-001'))
df['filepath'] = df['filepath'].apply(lambda x: x.replace(code_root, ""))
df.to_csv("output/code_search_openai-python.csv", index=False)
df.head()

from openai.embeddings_utils import cosine_similarity

def search_functions(code_query, n=2, pprint=True,n_lines=100):
    embedding = get_embedding(code_query, engine='code-search-babbage-text-001')
    df['similarities'] = df.code_embedding.apply(lambda x: cosine_similarity(x, embedding))

    res = df.sort_values('similarities', ascending=False).head(n)
    if pprint:
        for r in res.iterrows():
            print(r[1].filepath+":"+r[1].function_name + "  score=" + str(round(r[1].similarities, 3)))
            print("\n".join(r[1].code.split("\n")[:n_lines]))
            print('-'*70)
    return res 

# search_functions("stipulations/_title/", n=1, pprint=True, n_lines=100)


def get_line(path, line_number):
    # this gets the line before the line number, the line number, and the line after the line number
    # then puts them all together and returns them
    with open(path) as f:

        data = f.readlines()
    number_above = line_number - 4
    number_two_above = line_number - 4
    number_below = line_number + 1
    number_two_below = line_number + 4
    data[line_number] = data[line_number] + "\n"

    # lines between number_two_above and number_two_below
    lines = data[number_two_above:number_two_below]
    return "".join(lines)
    
    

    

        
        
    
    
    


            
    
        
    

line = get_line('/Users/canyonsmith/Desktop/enium/stipulations/_title/verify_title.py', input("Enter the line number: "))
path = '/Users/canyonsmith/Desktop/enium/' + input("Enter the path: ")
line = get_line(search_dir, int(input("Enter the line number: ")))
print(line)
search_functions(line, n=1, pprint=True, n_lines=1000)
# return the name of the function
function_name = search_functions(line, n=1, pprint=True, n_lines=1000)['function_name']
# print(function_name)
'''1    handler
Name: function_name, dtype: object
'''
# just return the name of the function
function_name = function_name.to_string(index=False)
print(function_name)
file_name = search_functions(line, n=1, pprint=True, n_lines=1000)['filepath']
command = "format this"

replace_function(function_name, file_name, edit_code(get_function_code(function_name, file_name),command))
