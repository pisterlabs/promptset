import sys, os
sys.path.append(os.getcwd())
from memory.memory import Memory
from utils.llm_utility import *
from langchain.vectorstores.chroma import Chroma
from langchain.docstore.document import Document
from evaluator import *

#__________________________________________________________________________________________________________________________

tool_database = Chroma(embedding_function = embedding_func, persist_directory= 'database/tool_mistake_db' , 
                                relevance_score_fn='similarity_search_with_score')
tool_mistake_memory = Memory(k=2, vector_db=tool_database)

#__________________________________________________________________________________________________________________________

def build_tool_experience(correct_tool, llm_tool):
    
    # try:
    response, analogy, correct_arguments = validate(correct_tool, llm_tool)
    # except:
        # response, analogy, correct_arguments = self.true_tools[self.tool_count] == output.tool, " ", " "

    if response is not True:
        print("\033[91m {}\033[00m" .format('Tool Arguments are not correct... (tool_memory)'))
        print("\033[91m {}\033[00m" .format('Staging tool experience in Memory... (tool_memory)'))
        experience = analogy
        metadata = {
            'tool_name': llm_tool[0]['tool_name']
        }
        doc = Document(page_content=experience , metadata=metadata)
        tool_mistake_memory.stage(doc)
        
    return response, correct_arguments


def retrieve_tool_experience(tool_name:str, user_query:str):
    if(tool_mistake_memory.queue.empty() == False):
        tool_mistake_memory.push()
    filter = {
        'tool_name':tool_name,
    }
    print("\033[91m {}\033[00m" .format('\nPulling argument mistakes from tool memory... (tool_memory)'))
    tool_mistakes = tool_mistake_memory.pull(query=user_query,filter=filter) if user_query != '' else ''    
    return tool_mistakes

    


