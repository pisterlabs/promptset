#plotly_chart.py
import streamlit as st
import numpy as np
from random import randrange
import openai,boto3,urllib, requests
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from PIL import Image
import re

#blah

def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]

def split_df(df):
    if len(df) % 2 != 0:  # Handling `df` with `odd` number of rows
        df = df.iloc[:-1, :]
    df1, df2 =  np.array_split(df, 2)
    return df1, df2


def response1(base_prompt):
    openai.api_key=st.secrets["open_api_key"]
    base_prompt = (f"{base_prompt}")
    engine = "text-davinci-003"
    #engine = "text-curie-001"
    
    
    messages = [ { "role": "user", "content": base_prompt } ]
    response = openai.ChatCompletion.create( model="gpt-3.5-turbo", messages=messages )
    #print(response)
    return  response["choices"][0]["message"]["content"]



def get_write_response (base_prompt) :
    
    with st.spinner ( response_while ) :
        answer=response1(base_prompt)
        st.subheader (response_after)
        st.markdown(answer)



def draw_prompt(dropdowns, tabname, df_d):


    select = df_d.dropdownname.unique().tolist()[0]
    s_d = st.radio ( str (select) + " : ", dropdowns , key = "dropdowns" + str( tabname) + "1")
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    tab_button=st.button(button_name , key = tab_name + "1")
    base_prompt = df_d [df_d.dropdown == s_d].prompt.unique().tolist()[0]
    st.markdown ( "--------")
    if tab_button:
        get_write_response (base_prompt)


def draw_prompt2(dropdowns, tabname, df_d):

    df_d1, df_d2 = split_df (df_d)
    col1, col2 = st.columns (2)
    with col1:

        dropdowns_col1 = df_d1 [ df_d1.tasks == tab_name ].dropdown.tolist()
        select = df_d1.dropdownname.unique().tolist()[0]
        s_d = st.radio ( str (select) + " : ", dropdowns_col1 , key = "dropdowns" + str( tabname) + "1")
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    
    with col2:
        dropdowns_col2 = df_d2 [ df_d2.tasks == tab_name ].dropdown.tolist()

        select = df_d2.dropdownname.unique().tolist()[0]
        s_d = st.radio ( str (select) + " : ", dropdowns_col2 , key = "dropdowns" + str( tabname) + "2" )
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    
    tab_button_1=st.button(button_name , key = tab_name + "1")
    base_prompt = df_d [df_d.dropdown == s_d].prompt.unique().tolist()[0]
    st.markdown ( "--------")
    if tab_button_1:
        get_write_response (base_prompt)



## main

st.set_page_config(page_title="Draft it for Me",layout='wide')

#m = st.markdown("""
#<style>
#div.stButton > button:first-child {
#    background-color: #0099ff;
#    color:#ffffff;
#}
#div.stButton > button:hover {
#    background-color: #00fffg;
#    color:#ffffff;
#    }
#</style>""", unsafe_allow_html=True)


m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #0099ff;
    color:#ffffff;
}
</style>""", unsafe_allow_html=True)


footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: red;
text-align: center;
}
</style>
<div class="footer">
<p> feedback, feature requests, bugs to <a href="mailto:rkreddy@gmail.com">rkreddy@gmail.com</a>   --  AI assisted drafts for other roles, visit our <a href="https://www.draftitforme.com">homepage</a> </p>
</div>
"""

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.markdown(footer,unsafe_allow_html=True)


button_name = "Draft it for me !! "
response_while = "Right on it, it should be around 2-5 seconds ..."
response_after = "Here you go ...  "


url = 'https://worldopen.s3.amazonaws.com/prompts_product_marketing.csv'
r = requests.get(url, allow_redirects=True)

open('/tmp/df.csv', 'wb').write(r.content)

df = pd.read_csv ('/tmp/df.csv', encoding = 'cp1252')

role = df.job.unique().tolist()[0]

st.header ( role.strip() )

tab_list = df.tasks.unique().tolist()

tabs = [ str(x) for x in tab_list if x is not np.nan ]

tabs = st.tabs ( tabs )  


i=0
for tab in tabs :

    with tab :
        tab_name = tab_list[i]
        #st.write (tab_name)
        df_d = df [ df.tasks == tab_name ]
        dropdowns = df [ df.tasks == tab_name ].dropdown.tolist()
        #st.write (dropdowns)
        draw_prompt(dropdowns, tab_name, df_d)

        i = i + 1

def old () :


    standards_tab, iam_tab, audit_tab, monitor_tab, detect_vuln_tab, bp_tab, raf_tab, sec_q_tab, rs_tab = st.tabs([  "Author a standards policy", "Author a secure IAM policy", "Audit Systems and Users", "Monitor Logs and Alerts", "Detect Vulnerabilities", "Recommend a Best Practice", "Reduce Attack Surface", "Respond to security questions", "Keep an eye on Spend (follow $$$)"] )

    with standards_tab :

        standard=st.selectbox("Select the Standard :", ("NIST 800-53", "HIPAA", "PCI"))
        s_type = st.selectbox ( "Select the control: ", ("Access , Authentication Password Policy", "Access, Least Privilege Policy", "Data Classification Policy", "Data Encryption Policy", "Data Protection Policy", "Data Sanitization Policy" , "Data Backup Policy", "System Monitoring Policy" ) )

        #if task == "Write a policy" :
            #standard=st.selectbox("Select the Language of  the Solution:", ("NIST", "HIPAA", "PCI"))

        base_prompt = "write a replace:policy in compliance with replace:standard standard, generate a bulleted list of items with controls that must be followed. include purpose, scope and policy sections" 
        
        base_prompt = base_prompt.replace ( "replace:policy", s_type.strip() )
        base_prompt = base_prompt.replace ( "replace:standard", standard.strip() )
        

        #question=st.text_area("Input the Question Here")
        button=st.button(button_name)
        st.markdown ( "--------")
        if button:
            get_write_response (base_prompt)

    with iam_tab :

        s_type = st.selectbox ( "Select IAM Access type: ", ("cross account access", "single account access") )
        service=st.selectbox("Select the service:", ("s3", "sns"))

        #if task == "Write a policy" :
            #standard=st.selectbox("Select the Language of  the Solution:", ("NIST", "HIPAA", "PCI"))

        base_prompt = """ write a secure iam policy and a role for cross account access of aws service replace:s_type.  the resource is in account 'a' and you want to access from a role in account. . the role in account "a" should restrict the principal to a specific role called 'service-role'.  the Resource key should be limited to the resource called resource-acta.  the Action should be limited to specific actions and not * """
        
        base_prompt = base_prompt.replace ( "replace:s_type", service.strip() )
        

        #question=st.text_area("Input the Question Here")
        iam_button=st.button(button_name , key = 'iam_button')

        if iam_button:

            st.subheader("Principles")
            st.markdown(
            """
            - Resource in account A, role service-role wants to access the resource from account B
            - Resources must not have `*`
            - Principal must not have `*`
            - Actions should be limited
            """
            )

            get_write_response (base_prompt)
           
     
    with audit_tab:

        
        s_type = st.selectbox ( "Select the Audit ask : ", (" Audit AWS account for CIS benchmarks; write a script so I can automate the task, in Python" , " Generate the list of Okta users for access auditing; write a script so I can automate the ask, in Python" ) )


        if 'aws account for cis' in s_type :
            base_prompt = """ write a python script to get cis benchmark results from aws security hub, use boto3 securityhub client """
        if "Okta users" in s_type :
            base_prompt = """  write python code to get okta users over okta api, for each user, get the group information that the user is in and applications that the users have access to. return results in a dataframe  """

        else :
            base_prompt = s_type

        #question=st.text_area("Input the Question Here")
        audit_button=st.button(button_name, key = "audit-button")

        st.markdown ("-------")
        if audit_button:
            get_write_response (base_prompt)

     
    with monitor_tab:

        
        s_type = st.selectbox ( "Select the detection ask : ", ("Get new Security Hub alerts, in the last week; Write a script for it so I can automate the task, in Python", "Get Cloud Trail events in the last hour; Wwrite a script for so I can automate the task, in Python", "Get Okta security events in the past day; Write a secipt for it, so I can automate the task, in Python" ) )


        if s_type == "Security Hub Alerts in the last week" :
            base_prompt = """ python code to generate aws security hub new alerts in the last week .   use CreatedAt filter to pass the start and end times .  do not use Criteria , just use createdat . CreatedAt is a list.  return pandas dataframe of the findings """
        if "Okta security events" in s_type :
            base_prompt = """   python code to get okta security events over okta api, return results in a dataframe  """
        if "gitlab container scans" in s_type.lower():
            base_prompt = """  write python code to get gitlab container security scan over gitlab api, return results in a dataframe  """

        else :
            base_prompt = s_type

        #question=st.text_area("Input the Question Here")
        monitor_button=st.button(button_name, key = "monitor-button")

        st.markdown ("-------")
        if monitor_button:
            get_write_response (base_prompt)

    with detect_vuln_tab:

        
        s_type = st.selectbox ( "Select the detection ask : ", ("Get GitLab container security scans ; write a script for it in python, so I can automate the task", "Get AWS Inspector ECR vulnerability scans ; write a script so I can automate the task, in Python", "Get Tenable vulnerability scans ; write a script for it in python, so I can automate the task, in Python" ) )


        if s_type == "Security Hub Alerts in the last week" :
            base_prompt = """ python code to generate aws security hub new alerts in the last week .   use CreatedAt filter to pass the start and end times .  do not use Criteria , just use createdat . CreatedAt is a list.  return pandas dataframe of the findings """
        if "Okta security events" in s_type :
            base_prompt = """   python code to get okta security events over okta api, return results in a dataframe  """
        if "gitlab container scans" in s_type.lower():
            base_prompt = """  write python code to get gitlab container security scan over gitlab api, return results in a dataframe  """
        if "ecr scans" in s_type.lower():
            base_prompt = """  write python code to get aws inspector ecr security scan using boto3 api, return results in a dataframe  """
        if "tenable" in s_type.lower():
            base_prompt = """  write python code to get tenable security scan using tenable workbench api, return results in a dataframe  """



        else :
            base_prompt = s_type

        #question=st.text_area("Input the Question Here")
        detect_vuln_button=st.button(button_name, key = "detect-vuln-button")

        st.markdown ("-------")
        if detect_vuln_button:
            get_write_response (base_prompt)


    with bp_tab:
     
        s_type = st.selectbox ( "Select the best practice : ", ("Recommend best practices for cloudwatch logging in AWS", "Recommend best practices for encrypting customer Data in AWS" ) )
        if s_type == "Delete un attacged volumes in AWS" :
            base_prompt = """ best practices for storing customer data in s3 , encryption, retention and tagging. include best practices around not copying data from production, sanitize before copying """
            base_prompt = """ best practices for logging access logs into cloudwatch logs , time stamp , who , when what action, what object, access control for logs, log level, log retention """

        #question=st.text_area("Input the Question Here")
        bp_button=st.button(button_name, key = "bp_tab_button")
        st.markdown ("-------")
        

       
        if bp_button:
            #answer = base_prompt
            get_write_response (base_prompt)
                #answer=response1(base_prompt)
                #st.write (response_after)
                #st.code(answer)

    with raf_tab:
        s_type = st.selectbox ( "Select : ", ("Find and delete un attached volumes in AWS, write code for it so I can automate, in Python", "Find and delete un tagged resources; write code for it so I can automate the task, in Python") )
        if s_type == "Find and delete un attached volumes in AWS, write code for it so I can automate, in Python" :
            base_prompt = """ write python code to Find and delete un attached EBS volumes in AWS, return a pandas dataframe  """ 

        if 'un tagged' in s_type :
             base_prompt = """ write python code to Find and delete un tagged resources in AWS, return a pandas dataframe """


        #question=st.text_area("Input the Question Here")
        raf_tab_button=st.button(button_name, key = 'raf_tab')
        st.markdown ("-------")

        if raf_tab_button:
            get_write_response (base_prompt)
            #answer = base_prompt
            #with st.spinner ( response_while ) :
                #answer=response1(base_prompt)
                #st.write (response_after)

                #st.code(answer)

    with sec_q_tab :
        standard=st.selectbox("Select the Standard :", ("NIST 800-53", "HIPAA", "PCI", "Monetary Authority of Singapore Technology Risk Management"))
        standard_number = st.text_input("control ",  "enter control, ex: authenticator management IA(5)")

        #if task == "Write a policy" :
            #standard=st.selectbox("Select the Language of  the Solution:", ("NIST", "HIPAA", "PCI"))

        base_prompt = "write a response to security questions from a customer how we are compliant with security " +  standard + " standard control " + str (standard_number) + " . start with we take security seriously and put in a numbered list of implementations, in a professional tone" 
        

        #question=st.text_area("Input the Question Here")
        sec_q_button=st.button(button_name, key = 'sec_q_button')
        st.markdown ( "--------")
        if sec_q_button or "enter control" not in standard_number:
            with st.spinner ( response_while ) :
                answer=response1(base_prompt)
                st.subheader (response_after)
                st.write(answer)




    with rs_tab :

        s_type = st.selectbox ( "Select a spend facet: ", ( "Get current spend grouped by service, write a script for it so I can automate the task, in python" , "Get current spend grouped by account, write a script for it so I can automate the task, in python" , "get forecast for the next month, write a script for it so I can automate the task, in python") )

        #if task == "Write a policy" :
            #standard=st.selectbox("Select the Language of  the Solution:", ("NIST", "HIPAA", "PCI"))

        if "forecast" in s_type.lower():
            base_prompt = """ python script to find aws spend forecast for the current month, starting today ending a month from now use UNBLENDED COST for the metric on forecast and MONTHLY for Granularity, the start and end dates should be in string yyyy-mm-dd format, give results in a dataframe  """

        if "spend grouped by service" in s_type.lower ():
            base_prompt = """ python script to to find current aws spend per day for the last 10 days, group by service , give the results in a pandas dataframe . use json_normalize with errors='ignore' .  the start time argument to get_cost_and_usage format should be yyyy-MM-dd.  convert keys column in the dataframe to string before groupby """ 
        
        if "spend grouped by account" in s_type.lower () :
            base_prompt = """ write the following python script :  make a call to aws cost explorer api to get the current spend , pass groupby the Type DIMENSION and Key  LINKED_ACCOUNT into get_cost_and_usage() api call, use DAILY for Granularity
                the start time argument to get_cost_and_usage format should be in strftime yyyy-MM-dd format with start datetime 10 days back (use timedelta to get the date)  and end datetime of today. use UnblendedCost for the Metrics
                give the results in a pandas dataframe , call that df
                take the Groups column from the dataframe, Iterate through each element in the column , apply json_normalize on each element, call that df_normalized 
                add the TimePeriod column from df  dataframe to df_normalize (df_normalized = json_normalize(element)) . append resulting dataframe into a list and make a new dataframe out of the concatenated dataframes in the list
                take the dataframe from the previous step, convert 'Metrics.UnblendedCost.Amount' column into float.  take that dataframe, groupby 'TimePeriod' and 'Keys' , sum by 'Metrics.UnblendedCost.Amount'
                do not use python functions """

        #question=st.text_area("Input the Question Here")
        rs_button=st.button(button_name, key = "rs_tab_button")
        st.markdown ("-------")

        if rs_button:
            get_write_response (base_prompt)
            
