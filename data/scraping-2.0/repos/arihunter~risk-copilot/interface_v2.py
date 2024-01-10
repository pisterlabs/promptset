import streamlit as st 
from llama_agent import OpenAIAgent
import openai 
from langchain.agents.tools import Tool
from langchain.chat_models import ChatOpenAI
from llama_index.tools.function_tool import FunctionTool
import pandas as pd
import dateutil.parser as dparser
import os
from typing import List
from google.oauth2 import service_account
import gspread
from sklearn import ensemble
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime
import logging
import sentry_sdk
from sentry_sdk import set_level
from sentry_sdk import capture_message
import uuid
import globals_
from difflib import SequenceMatcher


#Global Variables 




def before_send(event, hint):
    #print(globals_.chat_id)
    event['fingerprint'] = ["logging"]
    return event

sentry_sdk.init(
    dsn=st.secrets["SENTRY_DSN"],
    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    # We recommend adjusting this value in production.
    traces_sample_rate=1.0,
    # Set profiles_sample_rate to 1.0 to profile 100%
    # of sampled transactions.
    # We recommend adjusting this value in production.
    profiles_sample_rate=1.0,
    before_send=before_send
)

set_level("info")

openai.api_key = st.secrets["OPENAI_API_KEY"]

@st.cache_resource(show_spinner=False)
def gsheets_connection():
  credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"],scopes=["https://www.googleapis.com/auth/spreadsheets","https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"])
  gc = gspread.authorize(credentials)
  sheet_url = st.secrets["private_gsheets_url"]
  sheet = gc.open_by_url(sheet_url)
  return sheet
feedbackSheet = gsheets_connection()

st.set_page_config(page_title="Stealth")
col1,col2,col3 = st.columns([1,1,1])
col2.title("Workbench")
st.divider()

#CSS 
css = r'''
    <style>
        [data-testid="stForm"] {border: 0px}
    </style>
'''

st.markdown(css, unsafe_allow_html=True)


#initialisation of session state variables
if "thumbs" not in st.session_state.keys():
	st.session_state.thumbs = False

if "generate" not in st.session_state.keys():
	st.session_state.generate = False

if "query" not in st.session_state.keys():
	st.session_state.query = ""

if "response" not in st.session_state.keys():
	st.session_state.response = ""

if "prompt" not in st.session_state.keys():
	st.session_state.prompt = ""

if "citations" not in st.session_state.keys():
	st.session_state.citations = []

dataset_keys = ["lms","credit-decisioning","location"]
for key in dataset_keys:
  if key not in st.session_state :
    st.session_state[key] = False



# agent tools definitions

def gpt_helper(query:str,context:str) -> str:
  SYSTEM_PROMPT = f"{context}"
  model_id = "gpt-4"
  response = openai.ChatCompletion.create(
    model=model_id,
    messages=[
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":query}
    ],
    max_tokens=3000
  )
  return response["choices"][0]["message"]["content"]

output_formatting_prompt = f"""I need you to answer the user's query using the given context. The response to the query is certain to be in the context. Go carefully through the query and context and just return the answer, nothing else. Dont make anything up. Dont do any calculations on your end. Do not assume any denomination for the requested metrics in the query. Now using the given context answer the query. Context: \n"""

# #====reading all my datasets========
credit_decisioning_df = pd.read_csv("credit-decisioning_data.csv")
location_df = pd.read_csv("location_data.csv")
master_df_dict = {"bureau_data" : credit_decisioning_df, "location_data" : location_df}
master_col_dict = {"bureau_data" : credit_decisioning_df.columns, "location_data" : location_df.columns}

def pick_data_set(prompt : str)-> str:
  pick_data_set_prompt = """
  I will provide you with a user query, you have to analyse the query carefully and identify the feature name mentioned in the query. 
  If you're able to identify a feature from the user query you have to respond with the feature name else NO.
  Here are some examples for you.

  Example 1:
  Query: do the risk profiling of my portfolio using the bureau score
  Response: bureau score

  Example 2:
  Query : use the p_dist_gc_500 to do the risk profiling for year 2022 
  Response: p_dist_gc_500
  
  Example 3:
  Query : calculate NPA for year 2022
  Response: NO.
  """
  col_name_extracted = gpt_helper(prompt,pick_data_set_prompt)
  if col_name_extracted == "NO":
    return "There was insufficent information to do the analysis. Ask the user to give feature name in the query. Do not make up any random metrics by yourself"
  for key in master_col_dict:
    col_list = master_col_dict['key']
    for col in col_list:
      score = SequenceMatcher(None, col_name_extracted, col).ratio()
      if(score > 0.7):
        dataset_name = key
        col_name = col
        break
  return dataset_name, col_name

def calculate_risk_metrics(start_dt:str,end_dt:str) -> float:
  """
  This function is used to calculate the risk performance metrics of a lending portfolio, for a given time period. Risk performance metrics include the following:

  Disbursal amount or the loan amount which is defined as the loan amount disbursed.
  Defaulted amount which is defined as the loan amount which is defaulted or the amount which is due.
  Number of loans disbursed which is defined as the number of people approved for a loan.
  Average ticket size which is defined as the average value of the loan amount disbursed. It is calculated as a ratio of the disbursal amount to number of loans disbursed.
  NPA(non-performing asset) which is defined as the percentage of disbursed amount which is due or defaulted.

  """
  dataset_idx=[0]
  for idx in dataset_idx:
    if st.session_state[dataset_keys[idx]] == False:
      return "Sufficient data not available !, please provide all the required data."


  # input health check
  dateCheckPrompt = """
  I will provide you with a user query, you have to analyse the query carefully and identify if there is a complete time period mentioned in the query. 
  If you're able to identify all date elements from the user query you have to respond with YES else NO.
  Here are some examples for you.

  Example 1:
  Query: Calculate defaulted amount for the third financial quarter of 2021.
  Response: YES

  Example 2:
  Query : could you provide me the NPA calculation of the year 2022. I need it for each quarter individually
  Response: YES
  
  Example 3:
  Query : could you provide me the NPA calculation of the first quarter
  Response: NO.
  """
  
  dateCheck = gpt_helper(prompt,dateCheckPrompt)
  if dateCheck == "NO":
    return "There was insufficent date information to do the calculations. Ask the user to give complete date information in the query. Do not make up any random metrics by yourself."

  lms_df = pd.read_csv("lms_data.csv")
  lms_df["due_date"] = lms_df["due_date"].apply(lambda x: dparser.parse(x.split(" ")[0], dayfirst=True))
  lms_df_filtered = lms_df[(lms_df['due_date']) >= dparser.parse(start_dt, dayfirst=False)]
  lms_df_filtered = lms_df_filtered[(lms_df_filtered['due_date']) <= dparser.parse(end_dt, dayfirst=False)]
  lms_df_filtered["defaulted_amount"] = lms_df_filtered["is_default"] * lms_df_filtered["loan_amount"]
  defaulted_amount = lms_df_filtered['defaulted_amount'].sum()
  disbursal_amount = lms_df_filtered['loan_amount'].sum()
  number_of_loans_disbursed = lms_df_filtered['loan_amount'].count()
  average_ticket_siZe = disbursal_amount/number_of_loans_disbursed
  portfolio_npa = defaulted_amount / disbursal_amount
  context = {'defaulted_amount' : round(defaulted_amount,2), 'disbursal_amount' : round(disbursal_amount,2), 'number_of_loans_disbursed' : round(number_of_loans_disbursed,2), 'portfolio_npa' : round(portfolio_npa,2)}
  # response quality check
  example_context1 = {'defaulted_amount': 189256.01, 'disbursal_amount': 1020599.01, 'number_of_loans_disbursed': 4, 'portfolio_npa': 0.19}
  example_context2 = {'defaulted_amount': 1687413.0, 'disbursal_amount': 9991314.34, 'number_of_loans_disbursed': 45, 'portfolio_npa': 0.17}
  responsePrompt = f"""
  I need you to answer the users query using the given context. 
  The response to the query is certain to be in the context.
  Go carefully through the query and context and just return the answer, nothing else.
  Dont make anything up. Dont do any calculations on your end. Do not assume any denomination for the requested metrics in the query. 
  Here are some examples for you.

  Example 1:
  Context : {example_context1}
  Query: Calculate defaulted amount for the third financial quarter of 2021.
  Response: The defaulted amount for the third financial quarter of 2021 is 189256.01.

  Example 2:
  Context : {example_context2}
  Query : What is the NPA metric for the time period "01/04/2021" to "31/08/2021"
  Response: The NPA metric for the time period "01/04/2021" to "31/08/2021" is 0.17.

  Now using the given context answer the query.
  Context:
  {str(context)}
  """
  # responsePrompt = output_formatting_prompt
  # responsePrompt += str(context)
  capture_message(f"The values calculated are {context}")
  response = gpt_helper(prompt,responsePrompt)
  capture_message(f"Gpt helper response for risk metric function output formatting {response}")
  return response

def calculate_bureau_metrics(start_dt:str,end_dt:str) -> str:
  """This function is used to calculate various metrics related to the bureau score of the borrower. People who do not have a bureau score or have not taken a loan before are called New-To-Credit or NTC. People who have a bureau score or have taken a loan before are referred to as non-NTC. Using the bureau score the function can calculate the following:
      ntc_count which is the number of NTC in the portfolio
      non_ntc_count which is the number of non-ntc in the portfolio
      ntc_npa which is the bad-rate or npa of NTC in the portfolio
      non_ntc_npa which is the bad-rate or npa of non-NTC in the portfolio
      avg_bureau_defaulters is the average bureau scores of defaulters
      avg_bureau_non_defaulters is the average bureau scores of non-defaulters
      avg_ticket_siZe_defaulters is the average loan ticket siZe of defaulters
      avg_ticket_siZe_non_defaulters is the average loan ticket siZe of non-defaulters"""
  
  dataset_idx=[0,1]
  for idx in dataset_idx:
    if st.session_state[dataset_keys[idx]] == False:
      return "Sufficient data not available !, please provide all the required data."
  
  lms_df = pd.read_csv("lms_data.csv")
  credit_decisioning_df = pd.read_csv("credit-decisioning_data.csv")
  lms_df["due_date"] = lms_df["due_date"].apply(lambda x: dparser.parse(x.split(" ")[0], dayfirst=True))
  lms_df_filtered = lms_df[(lms_df['due_date']) >= dparser.parse(start_dt, dayfirst=False)]
  lms_df_filtered = lms_df_filtered[(lms_df_filtered['due_date']) <= dparser.parse(end_dt, dayfirst=False)]
  lms_df_filtered["defaulted_amount"] = lms_df_filtered["is_default"] * lms_df_filtered["loan_amount"]
  credit_decisioning_df = credit_decisioning_df.drop_duplicates()
  credit_decisioning_lms_df = pd.merge(lms_df_filtered, credit_decisioning_df, left_on = ["user_id"], right_on = ["user_id"], how = "left")
  bureau_score_col = 'bureau_score' #====this has to be detected from bureau data columns list=====
  ntc_df = credit_decisioning_lms_df[credit_decisioning_lms_df[bureau_score_col].isnull()]
  non_ntc_df = credit_decisioning_lms_df[~(credit_decisioning_lms_df[bureau_score_col].isnull())]
  ntc_count = ntc_df['user_id'].nunique()
  total = credit_decisioning_lms_df['user_id'].nunique()
  non_ntc_count = total - ntc_count
  ntc_npa = ntc_df['defaulted_amount'].sum()/ntc_df['loan_amount'].sum()
  non_ntc_npa = non_ntc_df['defaulted_amount'].sum()/non_ntc_df['loan_amount'].sum()
  avg_bureau_defaulters = credit_decisioning_lms_df[credit_decisioning_lms_df['is_default'] == 1][bureau_score_col].mean()
  avg_bureau_non_defaulters = credit_decisioning_lms_df[credit_decisioning_lms_df['is_default'] == 0][bureau_score_col].mean()
  avg_ticket_siZe_defaulters = lms_df[lms_df['is_default'] == 1]['loan_amount'].mean()
  avg_ticket_siZe_non_defaulters = lms_df[lms_df['is_default'] == 0]['loan_amount'].mean()
  context = {'ntc_count' : ntc_count, 'non_ntc_count' : non_ntc_count, 'ntc_npa' : ntc_npa,'non_ntc_npa' : non_ntc_npa, 'total_users' : total, 'avg_bureau_defaulters' : avg_bureau_defaulters, 'avg_bureau_non_defaulters' : avg_bureau_non_defaulters, 'avg_ticket_siZe_defaulters': avg_ticket_siZe_defaulters, 'avg_ticket_siZe_non_defaulters' : avg_ticket_siZe_non_defaulters}
  responsePrompt = output_formatting_prompt 
  responsePrompt += "\n" + str(context)
  capture_message(f"The values calculated are {context}")
  response = gpt_helper(prompt,responsePrompt)
  capture_message(f"Gpt helper response for risk metric function output formatting {response}")
  capture_message(f"The values calculated are {response}")
  return str(response)

def bin_df(df:pd.DataFrame,col_name:str,number_of_bins:int=5) -> pd.DataFrame:
  col_group_name = col_name + "_groups"
  df[col_group_name] = pd.qcut(df[col_name], number_of_bins)
  return df


def risk_profiling(start_dt:str,end_dt:str) -> str:
  """
  This function is used to do risk profiling for borrowers using a indicator, for a given time period.
  """
  # start_dt = st.text_input("Start Date")
  # end_dt = st.text_input("End Date")
  dataset_idx=[0,1]
  for idx in dataset_idx:
    if st.session_state[dataset_keys[idx]] == False:
      return "Sufficient data not available !, please provide all the required data."
  #sanity check
  dateCheckPrompt = "I will provide you with a user query, you have to analyse the query carefully and identify if there is a time period mentioned in the query. You have to respond only with YES or NO depending on the query."
  dateCheck = gpt_helper(prompt,dateCheckPrompt)
  if dateCheck == "NO":
    return "There was insufficent date information to do the calculations. Ask the user to give complete date information in the query. Do not make up any random metrics by yourself."
  lms_df = pd.read_csv("lms_data.csv")
  #===picking the dataset given the prompt=========
  dataset_name, col_name = pick_data_set(prompt)
  dataset_df = master_df_dict[dataset_name] #======how to handle this @Arihant??======
  credit_decisioning_df = pd.read_csv("credit-decisioning_data.csv")
  lms_df["due_date"] = lms_df["due_date"].apply(lambda x: dparser.parse(x.split(" ")[0], dayfirst=True))
  lms_df_filtered = lms_df[(lms_df['due_date']) >= dparser.parse(start_dt, dayfirst=False)]
  lms_df_filtered = lms_df_filtered[(lms_df_filtered['due_date']) <= dparser.parse(end_dt, dayfirst=False)]
  lms_df_filtered["defaulted_amount"] = lms_df_filtered["is_default"] * lms_df_filtered["loan_amount"]
  credit_decisioning_df = credit_decisioning_df.drop_duplicates()
  credit_decisioning_lms_df = pd.merge(lms_df_filtered, credit_decisioning_df, left_on = ["user_id"], right_on = ["user_id"], how = "left")
  credit_decisioning_lms_df = bin_df(credit_decisioning_lms_df, col_name, 5)
  col_group_name = col_name + "_groups"
  grouped_df = credit_decisioning_lms_df.groupby(col_group_name, dropna = False).agg({'user_id' : 'count','defaulted_amount' : 'sum', 'loan_amount' : 'sum'}).reset_index()
  grouped_df["npa"] = grouped_df["defaulted_amount"]/ grouped_df["loan_amount"]
  grouped_df["fraction_of_users"] = grouped_df["user_id"]/grouped_df["user_id"].sum()
  return grouped_df.to_string()

def calculate_top_features(external_data_lms_df, labels, num_fts = 10):
  features = external_data_lms_df
  gbm = ensemble.GradientBoostingRegressor()
  gbm.fit(features,labels)
  feature_importance = gbm.feature_importances_
  score_with_idx = [] 
  for idx,score in enumerate(feature_importance):
    score_with_idx.append((score,idx))
  sortedFeatures = sorted(score_with_idx)
  sortedFeatures.reverse()
  #===presently taking top 10 features only=====
  sortedFeatures = sortedFeatures[:num_fts]
  sortedIdx = [x[1] for x in sortedFeatures]
  top_fts = [features.columns[x[1]] for x in sortedFeatures]
  # top_fts = [features.columns[x] for x in sortedIdx]
  # df_with_top_fts = features.iloc[:,sortedIdx]
  # top_fts = list(df_with_top_fts.columns)
  # print(top_fts_v1)
  return str(top_fts)

def evaluate_data_source():
  """
  This function is used to evaluate an external data source and test it's performance in predicting the behavior of a risk portfolio. 
  The external data source will have a number of input columns called features, and this function will calculate the most important or powerful features which can add most value in building my underwriting logics or risk model. 
  Here top_fts has a list of top 10 features.
  """
  #hardcoding external data source here @Arihant - please make the change for the user to i/p the data source
  dataset_idx=[2]
  for idx in dataset_idx:
    if st.session_state[dataset_keys[idx]] == False:
      return "Sufficient data not available !, please provide all the required data."
  external_data = pd.read_csv("location_data.csv")
  external_data.dropna(inplace=True)
  labels = external_data["dep_var"]
  external_data.drop(["dep_var","address"],axis=1,inplace=True)
  top_fts = calculate_top_features(external_data,labels)
  #here both top_fts and response are to be processed wrt to a gpt_helper function. Here we might also need to output graphical trends
  return top_fts



#Agent creation
RiskProfileTool = FunctionTool.from_defaults(fn=risk_profiling)
RiskMetricsTool = FunctionTool.from_defaults(fn=calculate_risk_metrics)
BureauMetricsTool = FunctionTool.from_defaults(fn=calculate_bureau_metrics)
EvaluateDataTool = FunctionTool.from_defaults(fn=evaluate_data_source)



#Tools Llamaindex
llamaTools = [RiskMetricsTool,RiskProfileTool, BureauMetricsTool,EvaluateDataTool]
agentLlama = OpenAIAgent.from_tools(llamaTools)



#UI starts

def UploadedFileCallback(displayText:str):
  print(displayText)

with st.sidebar:

  #lms data
  LMS_DATA = st.file_uploader("LMS_DATA",type="csv",key="data_1")
  if LMS_DATA is not None:
    lms_path = "lms_data.csv"
    with st.spinner("Uploading"):
      st.session_state["lms"] = True
      pd.read_csv(LMS_DATA).to_csv(lms_path)


  CREDIT_DECISIONING_DATA = st.file_uploader("BUREAU_DATA",type="csv",key="data_2")
  if CREDIT_DECISIONING_DATA is not None:
    bureau_path = "credit-decisioning_data.csv"
    with st.spinner("Uploading"):
      st.session_state["credit-decisioning"] = True
      pd.read_csv(CREDIT_DECISIONING_DATA).to_csv(bureau_path)



  # COLLECTION_DATA = st.file_uploader("COLLECTION_DATA",type="csv",key="data_3")
  # if COLLECTION_DATA is not None:
  #   collection_path = "collection_data.csv"
  #   with st.spinner("Uploading"):
  #     st.session_state["collection"] = True
  #     pd.read_csv(COLLECTION_DATA).to_csv(collection_path)



  LOCATION_DATA = st.file_uploader("LOCATION_DATA",type="csv",key="data_4")
  if LOCATION_DATA is not None:
    location_path = "location_data.csv"
    with st.spinner("Uploading"):
      st.session_state["location"] = True
      pd.read_csv(LOCATION_DATA).to_csv(location_path)





  # DatasetOption = st.selectbox("Choose the Dataset",("LMS","Credit-Decisioning","Collection","Location"))
  
  
  # #Data Upload
  # if st.session_state[str(DatasetOption.lower())] == True:
  #   st.warning("Uploading now will update the dataset")
  #   displayText = "Updating the " + DatasetOption + "dataset"
  #   UploadedFile = st.file_uploader("Update the file here",type="csv",on_change=UploadedFileCallback,args=[displayText])
  #   if UploadedFile is not None:
  #     dataset_path = str(DatasetOption.lower()) + "_data.csv"
  #     string_path = str(DatasetOption.lower()) + "_data.txt"
  #     with st.spinner("Updating"):
  #       pd.read_csv(UploadedFile).to_csv(dataset_path)
  # else:
  #   displayText = "Uploading the " + DatasetOption + "dataset"
  #   UploadedFile = st.file_uploader("Upload the file here",type="csv",on_change=UploadedFileCallback,args=[displayText])
  #   if UploadedFile is not None:
  #     st.session_state[str(DatasetOption.lower())] = True
  #     dataset_path = str(DatasetOption.lower()) + "_data.csv"
  #     string_path = str(DatasetOption.lower()) + "_data.txt"
  #     with st.spinner("Uploading"):
  #       pd.read_csv(UploadedFile).to_csv(dataset_path)


userEmail = st.experimental_user.email

def find_last_filled_row(worksheet):
  return len(worksheet.get_all_values()) + 1

def insert_data_into_sheet(dataframe):
  worksheet = feedbackSheet.get_worksheet(0)  # Replace 0 with the index of your desired worksheet
  values = dataframe.values.tolist()
  last_filled_row = find_last_filled_row(worksheet)
  worksheet.insert_rows(values, last_filled_row)

#Callbacks
def ResponseCallback(prompt:str,response:str,_type:str):
  st.session_state.thumbs = True
  curr_time = datetime.now()
  date = curr_time.strftime("%m/%d/%Y")
  time = curr_time.strftime("%H:%M:%S")
  feedbackDict = {"Prompt":[prompt],"Response":[response],"Feedback":[_type],"UserId":[userEmail],"Date":[date],"Time":[time]}
  feedbackDf = pd.DataFrame.from_dict(feedbackDict)
  insert_data_into_sheet(feedbackDf)

def SubmitCallback():
  st.session_state.thumbs = False
  st.session_state.generate = True
 
def ClearCallback():
  st.session_state.query = ""
  st.session_state.generate = False
  st.session_state.thumbs = True


def BestPromptsCallback(query:str):
  st.session_state.prompt = query
  st.session_state.generate = True
  st.session_state.query = query

@st.cache_resource(show_spinner=False)
def get_response(query:str) -> str:
  response,citations = agentLlama.chat(query)
  st.session_state.citations = citations
  return str(response)


global prompt
prompt = st.text_area("Enter Here",key="query")
st.session_state.prompt = str(prompt)
col1,col2,col3 = st.columns([3,2,1],gap="large")
with col1:
  clear = st.button("Clear",on_click=ClearCallback,type="primary")
with col3:
  submit = st.button("Submit",on_click=SubmitCallback,type="primary")
  
  
st.markdown("")
if st.session_state.generate:
	with st.status("Generating response") as status:
		response = get_response(st.session_state.prompt)
		st.session_state.response = response
		for item in st.session_state.citations:
			st.write(item)
		status.update(label="How was the response generated?",expanded=True)
	placeholder = st.empty()
	st.write(f'<font size="4">{st.session_state.response}</font>',unsafe_allow_html=True)
	with placeholder.container():
		relevantCol1,relevantCol2,relevantCol3 = st.columns([0.8,0.1,0.1])
		if (st.session_state.thumbs == False):
			with relevantCol2:
				st.button(":thumbsup:",on_click=ResponseCallback,args=([str(prompt),str(response),"POSITIVE"]),disabled=False)
			with relevantCol3:
				st.button(":thumbsdown:",on_click=ResponseCallback,args=([str(prompt),str(response),"NEGATIVE"]),disabled=False)		


if st.session_state.generate == False:
  BestPrompts = ["What are the best features in my dataset to create a good underwriting model?","could you help me find the top 10 features from the location data which will impact my portfolio?","What was the NPA for the third financial quarter of 2022?"]
  col1,col2,col3 = st.columns([0.3,0.3,0.4])
  col1.button(BestPrompts[0],on_click=BestPromptsCallback,args=[BestPrompts[0]])
  col2.button(BestPrompts[1],on_click=BestPromptsCallback,args=[BestPrompts[1]])
  col3.button(BestPrompts[2],on_click=BestPromptsCallback,args=[BestPrompts[2]])
