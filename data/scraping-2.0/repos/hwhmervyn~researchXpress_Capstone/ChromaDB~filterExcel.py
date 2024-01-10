import sys, os
workingDirectory = os.getcwd()
sys.path.append(workingDirectory)

from filterConstants import chat_prompt, excel_parser, retry_prompt, output_fixer
from llmConstants import chat

import pandas as pd
import numpy as np
import textwrap
import plotly.graph_objects as go
from json.decoder import JSONDecodeError
from concurrent.futures import ThreadPoolExecutor
from langchain.callbacks import get_openai_callback

def correctFormatToJson(result_content, numTries, error_message):
  # Check if the number of tries has exceeded 3
  if numTries > 3:
     return None # Return None if too many retries, output will just be raw LLM output
  else:
    try:
      # if it hasn't been 3 tries yet, attempt to parse the result_content as JSON again
      jsonResult = output_fixer.parse_with_prompt(result_content, retry_prompt.format_prompt(error=error_message, output=result_content))
    except JSONDecodeError:
      # If a JSON decoding error occurs, do a recusive call to retry while increasing the counter for number of tries attempted
      jsonResult = correctFormatToJson(result_content, numTries+1, error_message) 
    except Exception: # Handle other exceptions by returning None, output will just be raw LLM Output
      jsonResult = None
    return jsonResult

def createTask(doi, title, abstract, query):
  # Check if both title and abstract are missing
  if pd.isna(title) and pd.isna(abstract):
    # Create a response for the case when both are missing
    jsonOutput = {
      'answer': 'Unsure',
      'explanation': 'Cannot tell when both Abstract and Title are missing'
    }
    return (doi, title, abstract, None, jsonOutput, 0, 0, 0)
  with get_openai_callback() as usage_info:
    # Create the prompt using the user input data and query
    request = chat_prompt.format_prompt(title=title, abstract=abstract, question=query).to_messages()
    result = chat(request) # send the request to LLM
    try:
      jsonOutput = excel_parser.parse(result.content) # Parse the result as JSON
    except JSONDecodeError as e:
      jsonOutput = correctFormatToJson(result.content, 1, str(e)) # Handle JSON decoding errors
    except Exception:
      jsonOutput = None # Handle other exceptions by returning only the raw LLM output, leaving the cleaned fields empty
    total_input_tokens = usage_info.prompt_tokens
    total_output_tokens = usage_info.completion_tokens
    total_cost = usage_info.total_cost
    return (doi, title, abstract, result.content, jsonOutput, total_input_tokens, total_output_tokens, total_cost)

def filterExcel(fileName, query):
   # Read data from an Excel file and drop rows with all missing values, then select specific columns
  df = pd.read_excel(fileName).dropna(how='all')[['DOI','TITLE','ABSTRACT']]
  executor = ThreadPoolExecutor(max_workers=2)
  futures = []
  for _, row in df.iterrows():
      doi, title, abstract = row
      # Create tasks for each row in the DataFrame and submit them to a thread pool
      futures.append(executor.submit(createTask, doi, title, abstract, query))
  return (executor, futures) # Return the thread pool executor and futures

def getOutputDF(dfOut):
    # Create new columns for PREDICTION and JUSTIFICATION FOR RELEVANCY
    json_exists = ~dfOut["jsonOutput"].isna()
    dfOut.insert(3,'PREDICTION', None, True)
    dfOut.loc[json_exists, 'PREDICTION'] = dfOut.loc[json_exists, "jsonOutput"].apply(lambda x: x.get('answer'))
    dfOut.insert(4,'JUSTIFICATION FOR RELEVANCY', None, True)
    dfOut.loc[json_exists, 'JUSTIFICATION FOR RELEVANCY'] = dfOut.loc[json_exists, "jsonOutput"].apply(lambda x: x.get('explanation'))
    dfOut = dfOut.drop('jsonOutput', axis=1) # Create new columns for PREDICTION and JUSTIFICATION FOR RELEVANCY
    return dfOut
  
def getYesExcel(df):
  yes_excel = df.copy()[df["PREDICTION"].str.lower() == "yes"]
  return yes_excel["PREDICTION"].tolist()

WRAPPER = textwrap.TextWrapper(width=80) 
COLOUR_MAPPING = {"Yes": "paleturquoise", "No": "lightsalmon", "Unsure": "lightgrey", np.nan: "white"}

#Add line breaks to the paragraphs
def add_line_breaks(text):
  new_text = "<br>" + "<br>".join(WRAPPER.wrap(text.strip())) 
  return new_text

#Clean the findings df
def clean_findings_df(uncleaned_findings_df):
  cleaned_findings_df = uncleaned_findings_df.copy()
  abstract_exists = ~cleaned_findings_df['ABSTRACT'].isna()
  #Add line breaks
  cleaned_findings_df.loc[abstract_exists,'Findings_Visualised'] = cleaned_findings_df.loc[abstract_exists,'ABSTRACT'].apply(lambda text: add_line_breaks(text))
  #Drop the Evidence column
  cleaned_findings_df = cleaned_findings_df.drop(columns = 'ABSTRACT')
  return cleaned_findings_df


#Generate a table visualisation
def generate_visualisation(cleaned_findings_df):

  layout = go.Layout(
    margin=go.layout.Margin(
      l=0, #left margin
      r=0, #right margin
      b=0, #bottom margin
      t=0,  #top margin
      pad=0
    )
  )

  fig = go.Figure(data=[go.Table(
    columnwidth = [150,150,430,70,150,150],
    header=dict(values=['DOI','Title','Abstract', 'Prediction', 'Justification for Relevancy', 'LLM Output'],
                fill_color='#203864',
                align='left', font=dict(color='white')),
    cells=dict(values=[cleaned_findings_df['DOI'], cleaned_findings_df['TITLE'],cleaned_findings_df['Findings_Visualised'],cleaned_findings_df['PREDICTION'],cleaned_findings_df['JUSTIFICATION FOR RELEVANCY'],cleaned_findings_df['LLM OUTPUT']],
               fill_color=['white','white','white',cleaned_findings_df['PREDICTION'].map(COLOUR_MAPPING),'white','white'],
               align='left')
    )
  ], layout=layout)

  return fig

# # FOR TESTING
# executor, futures = filterExcel("data/combined.xlsx",  "Is the article about Food.")
# from concurrent.futures import as_completed
# from tqdm import tqdm
# issues = []
# results = []
# with tqdm(total=len(futures)) as pbar:
#     for future in as_completed(futures):
#         row = future.result()
#         results.append(row)
#         pbar.update(1)

# dfOut = pd.DataFrame(results, columns = ["DOI","TITLE","ABSTRACT","llmOutput", "jsonOutput"])
# dfOut['prediction'] = dfOut["jsonOutput"].apply(lambda x: x['answer'])
# dfOut['justification_for_relevancy'] = dfOut["jsonOutput"].apply(lambda x: x['explanation'])
# dfOut.to_excel("test_output_pfa.xlsx")
# print()