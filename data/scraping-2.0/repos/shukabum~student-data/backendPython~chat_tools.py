from langchain.agents import Tool
from langchain.utilities.google_search import GoogleSearchAPIWrapper
from prompts import *
from llms import *
from parsers import *
from chains import *
from profile_retrivers import *
from neo4j_dir.creating_graph import *

# wolfram = WolframAlphaAPIWrapper()
search = GoogleSearchAPIWrapper()

template = '''

You are powerful at doing certain tasks like sorting, summarizing, formatting, etc.
You will be given a task with some instructions and you have to complete the task.

Make sure that you do not pollute the database with wrong information.

You need to format the respone in html in a conversational way, arrange the response in bulleted points 
and under major headings if possible. 

{input}

Take care that you do not pollute the data provided in response, by adding your own data.
Check that html syntax is correct.
Check that there are no back-ticks(`) in the output.

'''

prompt = PromptTemplate(
    input_variables=["input"],
    template=template,
)


task_tools = [
    # Tool(
    # name="Google Search",  
    # description="Use when the user wants to search something on web instead of relying on database. Use it when user insists on searching something on web.", 
    # func=search.run,  
    # return_direct=False, 
    # handle_tool_error=True,  
    # ),
    Tool(
    name = "Graph DataBase" , 
    description='''
        The tool can handle queries related to companies, job profiles, cgpa, ctc, venue, students, etc.
        The tool can handle queries like :
            1--> Which companies came for cgpa below 7?
            2--> What is the ctc offered by company X?
            3--> What ctc is offered for cgpa below 7?
            4--> Job profiles ofeered with job location as Bangalore?
            5--> List 20 companies which offered ctc above 30.
        
    ''',
    func = get_response,
    return_direct = True, 
    handle_tool_error=True,
    ),
    # Tool(
    #     name = 'LLM' , 
    #     description = '''
    #             This tool can handle variety of tasks like :
    #             1) summarizing the text
    #             2) sorting a list of objects/ numbers in a particular order.
    #             3) formatting the text in a particular way.
    #     ''' ,
    #     return_direct = True,
    #     handle_tool_error=True,
    #     func = LLMChain(llm=llm, prompt=prompt).run,
    # ) ,
    
]