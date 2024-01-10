# authentication
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
    
import pandas as pd
import os
import openai
from sqlalchemy import exc, create_engine, text as sql_text
import argparse
import gradio as gr

Message_Template_Filename = f'Template_MySQL-1.txt'
VDSDB_Filename =  "Question_Query_Embeddings-1.txt"
VDSDB = "Dataframe"

# for local modules
# sys.path.append(WD)
from nl2sql.NL2SQL_functions import Prepare_Message_Template, Run_Query
#from ChatGPT_Messages.src.lib.OpenAI_Func import Num_Tokens_From_String, OpenAI_Embeddings_Cost, OpenAI_Embeddings_Cost
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


def predict(Message, Verbose = False, Debug=False):
    if not hasattr(predict, "counter"):
        predict.counter = 0
    if not hasattr(predict, "Message_History"):
        predict.Message_History = Prepare_Message_Template(Template_Filename = Message_Template_Filename, Verbose=False, Debug=False)
        predict.VDSDB = VDSDB
        predict.VDS = VDS(VDSDB_Filename, Encoding_Base, Embedding_Model, Token_Cost, Max_Tokens) 
        predict.VDS.Load_VDS_DF(Verbose=False)

    # N shot examples
    if predict.counter == 0:
    # Request Question Embedding vector
        Question_Emb = predict.VDS.OpenAI_Get_Embedding(Text=Message, Verbose=Verbose)
        
    # Search Vector Datastore for similar questions
        rtn = predict.VDS.Search_VDS(Question_Emb, Similarity_Func = 'Cosine', Top_n=3)
        N_Shot_Examples = {'Question':rtn[1], 'Query':rtn[2]}
    # Append N Shot Examples to Message_History
        for i in range(len(N_Shot_Examples['Question'])):
            predict.Message_History.append({"role": "system", "name":"example_user", "content": N_Shot_Examples['Question'][i]})
            predict.Message_History.append({"role": "system", "name":"example_assistant", "content": N_Shot_Examples['Query'][i]})
  
    # Append Message (e.g. question)
    predict.Message_History.append({"role": "user", "content": Message})

    if Debug:
        print(f'predict.counter {predict.counter} {predict.Message_History}')
    
    # pass to LLM    
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages= predict.Message_History,
        temperature=0
      #  stream=True
    )

    Response = response['choices'][0]['message']['content']
    predict.Message_History.append({'role': 'assistant', 'content': Response})

    # now need to query DB
    df = Run_Query(Credentials=MYSQL_Credentials, Query=Response)

    # increment counter
    predict.counter += 1
    return Response, df


    
if __name__ == '__main__':
    p = argparse.ArgumentParser('Natural Language to SQL')
    p.add_argument('--Share', action='store_true', help=" Invoke Public Link ", default=False)

    args = p.parse_args()
    # Dataframe styles
    #styler = df.style.highlight_max(color = 'lightgreen', axis = 0)

    demo = gr.Interface(
        fn=predict,
        inputs = gr.Textbox(lines=4, placeholder="What is ...",label="Question"),
#        inputs=["text"],
 #       outputs=["text","dataframe"],
   #     outputs = ["text",gr.Dataframe()],
       outputs = [gr.Textbox(lines=4,label='Query',show_copy_button=True),gr.Dataframe(label="Result")],
        title="NL2SQL",
        description="Natural Language to SQL User Interface"
        ,allow_flagging="never"
    )

    demo.launch(share=args.Share, show_api=False, inbrowser=True) 


################################################################
#
def GPT_Search_N_Shot_Examples(Question, Verbose=False):
        # Request Question Embedding vector
        Question_Emb = VDS.OpenAI_Get_Embedding(Text=Question, Verbose=Verbose)
        
    # Search Vector Datastore for similar questions
        rtn = self._VDS.Search_VDS(self._Question_Emb, Similarity_Func = 'Cosine', Top_n=3)
        self._N_Shot_Examples = {'Question':rtn[1], 'Query':rtn[2]}
        return 0
