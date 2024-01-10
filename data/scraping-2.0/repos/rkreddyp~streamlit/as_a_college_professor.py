#from appsmills.streamlit_apps 
from helpers import openai_helpers
import streamlit as st
import numpy as np
from random import randrange
import openai,boto3,urllib, requests
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from PIL import Image
import re
## 




def streamlit_main (url) :


    button_name = "Draft it for me !! "
    response_while = "Right on it, it should be around 2-5 seconds ..."
    response_after = "Here you go ...  "


    #url = 'https://worldopen.s3.amazonaws.com/prompts_sales.csv'
    r = requests.get(url, allow_redirects=True)

    open('/tmp/df.csv', 'wb').write(r.content)

    df = pd.read_csv ('/tmp/df.csv', encoding = 'cp1252')

    role = df.job.unique().tolist()[0]

    st.header ( role.strip() )

    # tabs are the tasks
    tab_list = df.tasks.unique().tolist()

    tabs = [ str(x) for x in tab_list if x is not np.nan ]

    tabs = st.tabs ( tabs )  


    i=0
    for tab in tabs :

        with tab :
            tab_name = tab_list[i]
            #st.write (tab_name)
            df_d = df [ df.tasks == tab_name ]

            # these are the list of questions
            dropdowns = df [ df.tasks == tab_name ].dropdown.tolist()
            #st.write (dropdowns)
            openai_helpers.draw_prompt(dropdowns, tab_name, df_d)

            i = i + 1
           
streamlit_main ("https://worldopen.s3.amazonaws.com/prompts_professor.csv")


    
