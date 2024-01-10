# prototype chatbot withe four key functions:
# - generate tearsheet
# - query database and display data
# - query database and plot data
# - chat with vectorstore

# local imports
import tearsheet_utils as tshu
import email_utils as emut
from nl2sql import nl2sql_util

# system imports
import os
import openai
import argparse

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.memory import  ChatMessageHistory, ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage, HumanMessage
from langchain.schema.runnable import RunnablePassthrough
from langchain.tools import tool
from langchain.tools.render import format_tool_to_openai_function


from typing import Optional, List
from pydantic.v1 import BaseModel, Field

import gradio

# load vectordb
vectordb = tshu.create_or_load_vectorstore('data/chroma',
    tshu.load_persona_html(), 
    override=False)
 

# define chat_with_docs function
class ChatWithDocsInput(BaseModel):
    english_input: str = Field(..., description="Pass the entire user's natural language question unaltered into this parameter.")
    client_name: str = Field(..., description="Name of client to query for.")

@tool(args_schema=ChatWithDocsInput)
def chat_with_docs(english_input: str,  client_name: str) -> dict:
    # question: should document details go here or in the chat agent prompt?
    """
    Search the document store with the given query for the given client name. 

    Inputs:
        english_input - Should be user's original input unaltered.
        client_name - The name of the client to query for.

    The document store filters for the top documents given the query. Document
    types include:
    
    'equilar' - For details on stock or equity transactions, stock sold, annual compensation.
    'google' - For recent news articles listed on google.
    'linkedin' - For employment history, education, board memberships, and bio.
    'pitchbook' - For deals as lead partner, investor bio.
    'relsci' - For current or prior board memberships, top donations.
    'wealthx' - For individual and family net worth, interests, passions, hobbies.
    """

    client_name = ' '.join([w.capitalize() for w in client_name.split(' ')])

    # create filter and run query
    filter_ = tshu.create_filter(client_name, 'all')
    print(english_input, filter_)
    response = tshu.qa_metadata_filter(english_input, vectordb, filter_, top_k=5)

    #return f'called chat_with_docs for client {client_name}' #response
    return response


# define generate_and_send_tearsheet function
class GenSendTearsheetInput(BaseModel):
    client_name: str = Field(..., description="Name of client to generate tearsheet for.")
    email: str = Field(..., description="Email that will receive tearsheet this function sends.")

@tool(args_schema=GenSendTearsheetInput)
def gen_send_tearsheet(client_name: str, email: str) -> dict:
    """
    Generate the tearsheet for the given client_name. Send to the given email
    address. If the tearsheet already exists, it is read from disk rather than
    freshly remade.
    """

    client_name = ' '.join([w.capitalize() for w in client_name.split(' ')])

    # generate
    html, output_path = tshu.generate_tearsheet(client_name, vectordb, override=False)

    # todo: format / send email
    msg, success = emut.send_message(html, f'Your Tearsheet For {client_name}', email, verbose=False)

    return f'called gen_send_tearsheet for client {client_name} and recipient {email}: {success}' #response


# define generate_and_send_top3 function
class SendTop3Input(BaseModel):
    email: str = Field(..., description="Email that will receive report this function sends.")

@tool(args_schema=SendTop3Input)
def send_top3_email(email: str ='') -> dict:
    """
    Send the Top 3 alerts to the given email address. Only use this function if
    the user specifically requests an email about Top 3.
    """

    if email=='':
        return 'Email not sent because no email address was indicated. Use chat_with_db tool instead.'

    # generate
    table = chat_with_db.invoke({'english_input': 'show all of my top 3 recommendations'})

    html = f'''
    Dear Banker,<br><br>
    See below for your daily Top 3 recommendations.<br>
    {table}<br><br>
    Regards,<br>
    EQuABLE (Embedded Quant - AI Bank Lookup & Exploration)
    '''

    # todo: format / send email
    msg, success = emut.send_message(html, f'Your Daily Top 3 Recommendations', email, verbose=False)

    return f'called gen_send_top3 for recipient {email}: {success}' #response



# what clients do I have?
@tool
def list_my_clients() -> dict:
    """
    Look up all the client names according to metadata in the global vectorstore.
    """

    md = vectordb.get()['metadatas']
    return list(set([doc['client_name'] for doc in md]))



# define table_from_db
class ChatWithDB(BaseModel):
    english_input: str = Field(..., description="Pass the entire user's natural language question unaltered into this parameter.")

@tool(args_schema=ChatWithDB, return_direct=True)
def chat_with_db(english_input: str) -> dict:
    # question: should document details go here or in the chat agent prompt?
    """
    Search the database for banking information such as:
     - accounts and products,
     - client data (banker, address, employer name, entity type),
     - household relationships (clients grouped to the same entity)
     - client recommendations, including current Top 3 recommendations

    Given an input query, a function is called to convert the text to SQL and
    then run that query against the database.

    The tables in the database include:
     - account
     - address
     - balance
     - banker
     - client
     - date
     - employer
     - hh
     - naics
     - product
     - recommendations
     - relationship
     - transactions

    Inputs:
        query - Should be user's original English question unaltered.

    Output: The resulting dataframe is returned in HTML format.
    """

    sql, df = nl2sql_util.sql_to_df(english_input, return_sql=True)
    print(sql)
    print(df)
    return df.to_html(index=None)


# define plot_from_db

# define custom agent as class
class ChatAgent:
    '''Custom agent for using tools defined in this module.'''

    def __init__(self):
        # update this list of tools as more are added
        tools = [chat_with_docs, chat_with_db, gen_send_tearsheet, list_my_clients, send_top3_email]
        openai_functions = [format_tool_to_openai_function(f) for f in tools]

        # prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are the EQuABLE (Embedded Quant - AI Bank Lookup & Exploration) tool. You are a friendly AI tool that gives detailed responses to questions concerning bank data (relational databases, documents). You provide live answers as well as generate reports for distribution by email."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        # model
        model = ChatOpenAI(temperature=0).bind(functions=openai_functions)

        # memory
        memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
        
        # agent chain
        agent_chain = RunnablePassthrough.assign(
            agent_scratchpad= lambda x: format_to_openai_functions(x["intermediate_steps"])
        ) | prompt | model | OpenAIFunctionsAgentOutputParser()


        self.agent = AgentExecutor(agent=agent_chain, tools=tools, verbose=True, memory=memory)

    def run(self, user_input):
        '''Function that submits user input to chatbot.'''

        response = self.agent.invoke({"input": user_input})
        #repsonse is a dict with keys for input, output, chat_history

        return response['output']


if __name__ == '__main__':
    import chatbot1 as m

    """
    # todo: check right documents retrieved with simple query
    # invent llm call to tag question with document types and use those as
    # part of chat_with_docs: nested llm calls
    # response: yes, this is working correctly, verified with
    # vectordb.similarity_search and filter = {'client_name': 'Robert King'}

    tools = [m.chat_with_docs, m.gen_send_tearsheet]
    functions = [m.format_tool_to_openai_function(f) for f in tools]
    model = m.ChatOpenAI(temperature=0).bind(functions=functions)
    prompt = m.ChatPromptTemplate.from_messages([
        ("system", "You are helpful but sassy assistant"),
        ("user", "{input}"),
    ])
    chain = prompt | model | m.OpenAIFunctionsAgentOutputParser()

    # map tool names to callables 
    tool_map = {
        "chat_with_docs": m.chat_with_docs, 
        "gen_send_tearsheet": m.gen_send_tearsheet,
        }

    # test that questions map to the right functions
    # eval output with resultN.log
    result1 = chain.invoke({"input": "where does Robert King work?"})
    result2 = chain.invoke({"input": "what is Robert King's family net worth?"})
    result3 = chain.invoke({"input": "What deals as lead partner did Robert King do?"})
    result4 = chain.invoke({"input": "What recent news is there about Robert King?"})
    result5 = chain.invoke({"input": "Write a tearsheet about Robert King and send it to tweinzirl@gmail.com"})
    result6 = chain.invoke({"input": "Write a tearsheet about Julia Harpman and send it to tweinzirl@gmail.com"})
    result7 = chain.invoke({"input": "Write a tearsheet about Velvet Throat and send it to tweinzirl@gmail.com"})
    result8 = chain.invoke({"input": "Write a tearsheet about Jared Livinglife and send it to tweinzirl@gmail.com"})

    for i, result in enumerate([result1, result2, result3, result4, result5, result6, result7, result8]):
        observation = tool_map[result.tool].run(result.tool_input)
        print(f'\n{i}, {result.log} : {observation}\n')
    """

    # args
    parser = argparse.ArgumentParser(description='Start app. Optionally share to web.')
    parser.add_argument('--share', dest='share', action='store_true',
                        help='Share the app to the web (default: False).')

    args = parser.parse_args()


    # chatbot interface
    agent = m.ChatAgent()

    def agent_chat(query, history):
        '''
        Function to link gradio chat interface with langchain agent.
        Agent history is customized to match the chat history of the chat.
        This allows separate chats to take place simultaneously.
        Code based on https://www.gradio.app/guides/creating-a-chatbot-fast#a-langchain-example
        '''
        message_history = []
        for human, ai in history:
            message_history.append(HumanMessage(content=human))
            message_history.append(AIMessage(content=ai))

        agent.agent.memory.chat_memory = ChatMessageHistory(messages=message_history)

        return agent.run(query)

    CSS ="""
        .contain { display: flex; flex-direction: column; }
        .gradio-container { height: 100vh !important; }
        #component-0 { height: 100%; }
        #chatbot { flex-grow: 1; overflow: auto;}
    """  # custom css does not work to make chat taller

    demo = gradio.ChatInterface(fn=agent_chat,
        chatbot=gradio.Chatbot(elem_id="chatbot"),
        examples=["Where does client |NAME| work?",
                  "Send the tearsheet for client |NAME| to |EMAIL|",
                  "List my clients",
                  "Count my recommendations by report name",
                  "List my Top 3 recommendations"],
        title="EQuABLE: Embedded Quant - AI Bank Lookup & Exploration",
        retry_btn=None,
        undo_btn=None,
        clear_btn='Clear Chat History',
        css=CSS,
        )
    ### 
    demo.launch(share=args.share)


    #### todo: 
    #   x 1) update prompt in qa chain to always use the ocntext and ignore objections over not having access to personal information or recent news on google.
    # e.g., what is julia harpman doing these days?
    # - does julia harpman own any stock yet?

    # 2) email transcript of chat, may require agent as global variable

    # 3) customize dimensions, style of chatbot

    # good questions:
    #   - what is the employment history of Robert King as a markdown table?

    #   - summarize Robert King\'s board membership history. separate current from prior positions. format as a markdown table
    #   - list deals where client robert king has been lead partner
    #     -- who was robert king representing in that deal
    #     -- what was the size of the deal
    #     -- what is the status of the deal
    #     -- what other partners were involved in the deal

    #   - tell me about Velvet Throat's music career

    #   - tell chatbot what the query it should use is: for client robert king run query "print date for the article called `Robert King donates $1 million to a local charity`"
