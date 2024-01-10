from llmConstants import chat
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List
from json.decoder import JSONDecodeError

import pandas as pd
import textwrap
import json
import re 
from json.decoder import JSONDecodeError
import plotly.graph_objects as go

import sys, os
workingDirectory = os.getcwd()
chromaDirectory = os.path.join(workingDirectory, "ChromaDB")
costDirectory = os.path.join(workingDirectory, "cost_breakdown")
sys.path.append(costDirectory)
sys.path.append(chromaDirectory)

import chromaUtils
from cost_breakdown.update_cost import update_usage_logs, Stage

### Global Parameters ###
# Aesthetic parameters
COLOUR_MAPPING = {"Yes": "paleturquoise", "No": "lightsalmon", "Unsure": "lightgrey"}
# Text wrap for output in table
WRAPPER = textwrap.TextWrapper(width=145) #Creates a split every 160 characters

# Create a class for output parser
class Response(BaseModel):
    answer: str = Field(description= "Answer Yes or No in 1 word" )
    evidence: List[str] = Field(description="List 3 sentences of evidence to explain")

# Parser
OUTPUT_PARSER = PydanticOutputParser(pydantic_object=Response)

# Create the prompt based on prompt template + output parser
def create_prompt():
  mention_y_n_prompt_template = """
    [INST]<<SYS>>
    You are a psychology researcher extracting findings from research papers.
    If you don't know the answer, just say that you don't know, do not make up an answer.
    Use only the context below to answer the question.<</SYS>>
    Context: {context}
    Question: {question}
    Format: {format_instructions}
    """

  mention_y_n_prompt = PromptTemplate(
    template= mention_y_n_prompt_template,
    input_variables=["context", "question"],
    partial_variables={"format_instructions": OUTPUT_PARSER.get_format_instructions()})
  
  return mention_y_n_prompt

# Fix the output of the string for the json to detect that it's a dictionary
def fix_output(string, llm):
    prompt_template = """Convert the given string to a JSON object. 
      Format: ### {format_instructions} ###
      String to convert: ### {string} ###
    """
    prompt = PromptTemplate(template=prompt_template,
                              input_variables=["string"],
                              partial_variables={"format_instructions": OUTPUT_PARSER.get_format_instructions()})
    format_correction_chain = LLMChain(llm=llm, prompt=prompt)
    result = format_correction_chain.run(string)
    return result

# Correct output for the evidence if needed else return original
def check_evidence_format(result, llm):
  evidence = None
  try:
    result_dict = json.loads(result)
    evidence = result_dict['evidence']
  except JSONDecodeError:
    json_string = fix_output(result, llm)
    try:
      result_dict = json.loads(json_string)
      evidence = result_dict['evidence']
    except Exception:
      evidence = result
  except Exception:
    evidence = result
  return evidence

# Retrieve findings from the llm
def get_findings_from_llm(query, pdf_collection, specific_filename, mention_y_n_prompt, llm):
  # Create a Retrieval Chain
  qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                         chain_type="stuff",
                                         retriever= pdf_collection.as_retriever(search_type="similarity", search_kwargs={'k': 3, 'filter': {'fileName': specific_filename}}),
                                         chain_type_kwargs={"prompt": mention_y_n_prompt},
                                         return_source_documents=True)
  # Get the results
  result_dict = qa_chain({"query": query})
  result = result_dict['result']
  return result

# Queries the pdfs and outputs a dataframe
def get_findings_from_pdfs(pdf_collection, collection_name, query, mention_y_n_prompt, llm, progressBar1):
  # Get the unique filenames from the pdf collection
  unique_filename_lst = chromaUtils.getDistinctFileNameList(collection_name)
  total_num_articles = len(unique_filename_lst)
  # List to store yes or no
  yes_no_lst = []
  # List to store the evidence
  evidence_lst = []
  print(total_num_articles)

  # For progress bar
  PARTS_ALLOCATED_IND_ANALYSIS = 0.5
  numDone = 0
  progressBar1.progress(0, text=f"Analysing articles: {numDone}/{total_num_articles} completed...") 

  with get_openai_callback() as usage_info:
    for specific_filename in unique_filename_lst:
      # Get the findings using the LLM
      result = get_findings_from_llm(query, pdf_collection, specific_filename, mention_y_n_prompt, llm)
      # Check whether the pdf is related to the research question and update the lists accordingly
      if 'Yes' in result:
        yes_no_lst.append('Yes')
      else:
        yes_no_lst.append('No')

      # Get the evidence 
      evidence = check_evidence_format(result, llm)
      evidence_lst.append(evidence)

      numDone += 1
      progress = (float(numDone/total_num_articles) * PARTS_ALLOCATED_IND_ANALYSIS)
      progress_display_text = f"Analysing articles: {numDone}/{total_num_articles} completed..."
      progressBar1.progress(progress, text=progress_display_text)
    # Update the usage
    total_input_tokens = usage_info.prompt_tokens
    total_output_tokens = usage_info.completion_tokens
    total_cost = usage_info.total_cost
    update_usage_logs(Stage.PDF_FILTERING.value, query, total_input_tokens, total_output_tokens, total_cost)   

  # Output a dataframe
  uncleaned_findings_dict= {'Article Name': unique_filename_lst, 'Answer' : yes_no_lst, 'Evidence' : evidence_lst}
  uncleaned_findings_df = pd.DataFrame(uncleaned_findings_dict)
  return uncleaned_findings_df

# Add line breaks to the paragraphs
def add_line_breaks(text_list):
  new_text_list = []
  for text in text_list:
    # Add line breaks for easier viewing of output
    new_text = "<br>" + "<br>".join(WRAPPER.wrap(text.strip())) 
    new_text_list.append(new_text)
  return "".join(new_text_list)

# Clean the findings df
def clean_findings_df(uncleaned_findings_df):
  cleaned_findings_df = uncleaned_findings_df.copy()
  # Get the findings just paragraphs
  cleaned_findings_df['Findings'] = cleaned_findings_df['Evidence'].apply(lambda evidence_list : " ".join(evidence_list))
  # Add line breaks
  cleaned_findings_df['Findings_Visualised'] = cleaned_findings_df['Evidence'].apply(lambda evidence_list: add_line_breaks(evidence_list))
  # Drop the Evidence column
  cleaned_findings_df = cleaned_findings_df.drop(columns = 'Evidence')
  return cleaned_findings_df

# Generate table visualisation
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
    columnwidth = [50,50,300],
    header=dict(values=['<b>Article Name</b>', '<b>Answer</b>', '<b>Findings</b>'],
                fill_color='#203864',
                align='left', 
                font=dict(color='white')),
    cells=dict(values=[cleaned_findings_df['Article Name'], cleaned_findings_df['Answer'], cleaned_findings_df['Findings_Visualised']],
               fill_color=["white", cleaned_findings_df['Answer'].map(COLOUR_MAPPING), "white"],
               align='left')
    )
  ], layout=layout)

  return fig

# Get pdf filenames that mention the topic
def get_yes_pdf_filenames(cleaned_findings_df):
  yes_pdf = cleaned_findings_df.copy()[cleaned_findings_df["Answer"].str.lower() == "yes"]
  return yes_pdf['Article Name'].values.tolist()

def ind_analysis_main(query, collection_name, progressBar1):
  # Get the pdf collection
  pdf_collection = chromaUtils.getCollection(collection_name)

  # Initialise the prompt template
  mention_y_n_prompt = create_prompt()
  
  # Get findings from the pdfs
  uncleaned_findings_df = get_findings_from_pdfs(pdf_collection, collection_name, query, mention_y_n_prompt, chat, progressBar1)

  # Clean the findings
  cleaned_findings_df = clean_findings_df(uncleaned_findings_df)

  # Generate the visualisations
  fig = generate_visualisation(cleaned_findings_df)

  return cleaned_findings_df[["Article Name", "Answer", "Findings"]], fig