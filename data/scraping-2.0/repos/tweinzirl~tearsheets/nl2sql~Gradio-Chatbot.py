# authentication
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# Interface_Multiple-openai.py
import pandas as pd
import os
import openai
from sqlalchemy import exc, create_engine, text as sql_text
from dotenv import load_dotenv
import argparse
import gradio as gr
import random

Message_Template_Filename = 'nl2sql/Template_MySQL-1.txt'
VDSDB_Filename =  "nl2sql/Question_Query_Embeddings-1.txt" 
VDSDB = "Dataframe"

# for local modules
# import sys
# sys.path.append(WD)
from nl2sql.NL2SQL_functions import Prepare_Message_Template, Run_Query, Export_df
from nl2sql.lib_OpenAI_Embeddings import VDS

MYSQL_USER = os.getenv("MYSQL_USER", None)
MYSQL_PWD = os.getenv("MYSQL_PWD", None)
openai.api_key = OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
MYSQL_Credentials = {'User':MYSQL_USER,'PWD':MYSQL_PWD}

Embedding_Model = "text-embedding-ada-002"
Encoding_Base = "cl100k_base"
Max_Tokens = 250
Temperature = 0
Token_Cost = {"gpt-3.5-turbo-instruct":{"Input":0.0015/1000,"Output":0.002/1000},
                "gpt-3.5-turbo":{"Input":0.001/1000,"Output":0.002/1000},
                "text-embedding-ada-002":{"Input":0.0001/1000, "Output":0.0001/1000}}


################################################################
# Respond
def respond(message, chat_history, Verbose=False, Debug=False):
    # initialize attributes when Chat History is clear
    
    if chat_history == []:
        respond.Message_History = Prepare_Message_Template(Template_Filename = Message_Template_Filename, Verbose=False, Debug=False)
        respond.counter = 0
        respond.VDSDB = VDSDB
        respond.VDS = VDS(VDSDB_Filename, Encoding_Base, Embedding_Model, Token_Cost, Max_Tokens) 
        respond.VDS.Load_VDS_DF(Verbose=False)  

    if Verbose:
        print(f'{respond.counter} message {message} \n chat_history {chat_history}')

     # Find N shot examples on first question
    if respond.counter == 0:
    # Request Question Embedding vector
        Question_Emb = respond.VDS.OpenAI_Get_Embedding(Text=message, Verbose=Verbose)
        
    # Search Vector Datastore for similar questions
        rtn = respond.VDS.Search_VDS(Question_Emb, Similarity_Func = 'Cosine', Top_n=3)
        N_Shot_Examples = {'Question':rtn[1], 'Query':rtn[2]}
    # Append N Shot Examples to Message_History
        for i in range(len(N_Shot_Examples['Question'])):
            respond.Message_History.append({"role": "system", "name":"example_user", "content": N_Shot_Examples['Question'][i]})
            respond.Message_History.append({"role": "system", "name":"example_assistant", "content": N_Shot_Examples['Query'][i]})
  
    # Append Message (e.g. question)
    respond.Message_History.append({"role": "user", "content": message})

    # call LLM    
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages= respond.Message_History,
        temperature=0
      #  stream=True
    )

    # extract LLM response
    Response = response['choices'][0]['message']['content']

    # Append Message History
    respond.Message_History.append({'role': 'assistant', 'content': Response})

    # Append Chat History
    chat_history.append((message,f'{respond.counter} \n {Response}'))

    if Debug:
        print( f'respond.counter {respond.counter} size {len(respond.Message_History)}' )
        for i in respond.Message_History:
            print(i)

    # now need to query DB
    rtn_df = Run_Query(Credentials=MYSQL_Credentials, Query=Response)

    # increment counter
    respond.counter += 1
    return "", chat_history, rtn_df

################################################################
# Main

if __name__ == '__main__':
    p = argparse.ArgumentParser('Natural Language to SQL Chatbot')
    p.add_argument('--Share', action='store_true', help=" Share via public url ", default=False)
    p.add_argument('--InBrowser', action='store_true', help=" Start UI in a new tab", default=False)

    args = p.parse_args()

    css = """
    .gradio-container {background-color: black}
    .Chatbot label {background-color: teal !important}
    .Message label {background-color: teal !important}
    .Result {background-color: teal !important}
    .Clear {background: orange !important}
    .Export_Results {background-color: teal !important}
    """
    with gr.Blocks(css=css) as demo:
        gr.Markdown("<h2>Natural Language to SQL Chatbot</h2>")
        with gr.Row():
            with gr.Column():
                chatbot = gr.Chatbot(show_copy_button=True,
                                    label="Chatbot",
                                    layout='bubble',
                                    likeable=False,
                                    bubble_full_width=True,
                                    render_markdown=True
                                    ,container=True
                                    ,elem_classes="Chatbot")
                msg = gr.Textbox(label='Message', elem_classes="Message", placeholder="Message N2SQLChat ..." )
                clear = gr.ClearButton([msg, chatbot], value='Reset', size="sm", elem_classes="Clear")

            with gr.Column():
                result_df = gr.Dataframe(label=" ",interactive=False,elem_classes="Result")
                with gr.Accordion(label="Export Results",open=False,elem_classes="Export_Results"):
                    gr.Markdown("")
                    Filename = gr.Textbox(label='Filename',placeholder="Enter filename")
                    Export_msg = gr.Textbox(label="")
                    btn = gr.Button(size="sm",value="Export")
                    btn.click(Export_df, [result_df, Filename], Export_msg)
               # dataset = gr.Dropdown(
                    #choices=["stocks", "climate", "seattle_weather", "gapminder"],
               #     choices = list(result_df.columns) if hasattr(result_df, "column") else [],
                #    allow_custom_value=True
                #)
                #tmp = gr.Textbox(label='Tmp')
               # dataset.change(test, inputs=dataset, outputs=tmp)

        msg.submit(respond, [msg, chatbot],[msg, chatbot, result_df])


    demo.launch(share=args.Share, show_api=False, inbrowser=True) 


################################################################
#
