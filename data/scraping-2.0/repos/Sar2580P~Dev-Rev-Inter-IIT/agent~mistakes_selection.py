from langchain.prompts import PromptTemplate  
from utils.llm_utility import llm 
from langchain.chains import LLMChain
from typing import List, Union
from langchain.docstore.document import Document
import ast
import sys, os
sys.path.append(os.getcwd())
from utils.templates_prompts import MISTAKE_SELECTION
from memory.memory import mistake_memory

prompt = PromptTemplate(template=MISTAKE_SELECTION, input_variables=["input", "mistake"])
chain = LLMChain(llm=llm, prompt=prompt)

def choose_mistake(mistake, tool_task):
    return  chain.run({'input':tool_task,  'mistake': mistake})
   

def analyse(user_query, wrong_tool_name, tool_task):
    final_mistakes = [] 
    if tool_task == None:
        return final_mistakes
    
    print("\033[91m {}\033[00m" .format('analyse (mistake_selection)'))
    print("\033[91m {}\033[00m" .format('\tPulling mistakes from agent memory... (mistake_selection)'))

    filter = {
        'wrong_tool': wrong_tool_name,
    }
    # mistakes = mistake_memory.pull(query=user_query, filter=filter) if user_query != '' else ''
    mistakes = mistake_memory.pull(query=tool_task, filter=filter) if user_query != '' else ''
    mistaken_tool_set = set()
    
    if isinstance(mistakes , str) or mistakes == []:
        return 'No mistakes found  relevant to this query'
    
    i=0
    for mistake in mistakes:
        mistaken_tool = mistake.metadata['correct_tool']
        if not mistaken_tool in mistaken_tool_set :
            mistaken_tool_set.add(mistaken_tool)
            ans = choose_mistake(mistake.metadata['learning'] , tool_task=mistake.page_content)
            if '1' in ans:
                i+=1
                print("\033[91m {}\033[00m" .format('\tchosen_mistakes : {i} (mistake_selection)'.format(i=i)))
                # print("\033[93m {}\033[00m" .format(prompt.template.format(input=user_query , mistake=mistake.page_content)))
                final_mistakes.append(mistake)

    return final_mistakes

    # for mistake in mistakes:
    #     # mistaken_tool = mistake.metadata['correct_tool']
    #     # if not mistaken_tool in mistaken_tool_set:
    #     # mistaken_tool_set.add(mistaken_tool)
    #     # ans = choose_mistake(user_query=user_query , mistake=mistake.metadata['learning'])
    #     # if ans == 1:
    #     #     i+=1
    #     #     print("\033[91m {}\033[00m" .format('\tchosen_mistakes : {i} (mistake_selection)'.format(i=i)))
    #     #     print("\033[93m {}\033[00m" .format(prompt.template.format(input=user_query , mistake=mistake.page_content)))
    #     final_mistakes.append(mistake)
    # print("Final Mistakes = ", final_mistakes)
    # return final_mistakes
