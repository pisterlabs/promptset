######################################################
##### Generate Q & A from Docs for Fine Tuning #######
######################################################

"""
The following script can be used to create questions and answers for a set of documents (PDF, DOCX, CSV) to generate training data for fine tunning a LLM 
for your Q&A application. It leverages the OpenAI API to generate questions and answers from snippets of your document collection. The following are parameters
to keep in mind:

- chunk_size : dictates how long each snippet will be. Larger snippets will provide more context but will be more expensive (more tokens).
- sample_size: dictates how many Q&As will be generated. The script randomly pulls n-samples from your processed corpus. Larger n_samples means more questions 
             but will incur more cost.

"""

import os
import shutil
from dotenv import load_dotenv, find_dotenv

from PyPDF2 import PdfReader
import re
import docxpy
import random
import pandas as pd

from langchain.llms import OpenAI
from langchain import PromptTemplate, HuggingFaceHub, LLMChain

from helpers import *

"""
Specify API Keys
"""
OPENAI_API = os.getenv('OPENAI_API_KEY')

"""
Specify preprocessing

convert_txt: converts PDF and DOCX files to TXT files
chunk_txt: chunks the TXT files into smaller snippets

"""
convert_txt = False
chunk_txt = True

"""
Specify filepaths
"""

def get_filepath():
  """Returns the filepath of the directory"""
  filepath = os.path.dirname(os.path.realpath(__file__))
  return filepath

# Get filepaths
main_dir = get_filepath()
doc_dir = os.path.join(main_dir, 'docs')
clean_dir = os.path.join(main_dir, 'docs_clean')
out_dir = os.path.join(main_dir, 'output')

# Reset directory
if convert_txt:
    if os.path.exists(clean_dir):
        shutil.rmtree(clean_dir)
        os.makedirs(clean_dir)
    else:
       os.makedirs(clean_dir)

# Reset outputs
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
    os.makedirs(out_dir)
else:
   os.makedirs(out_dir)

"""
Text Processing

The files in PDF, docx and CSV in the 'docs' directory will be converted to .txt format and stored in the 'docs_clean' folder

"""

"""
Convert to TXT Preprocessing
"""
if convert_txt:
    
    text_ls = []

    """
    Iterate through all pdfs and docx files and convert them to text. 
    Save them to a list as a tuple with the document file location.
    """
    for i in os.listdir(doc_dir):
        filename = os.path.join(doc_dir, i)
        if re.search('.pdf', filename) is not None:
            print(filename)
            text = ""
            with open(filename, 'rb') as f:
                reader = PdfReader(f)
                for pg in reader.pages:
                    text += pg.extract_text()
                text = text.strip()
                #text = text.replace("\n", "")
                #text = text.replace("\t", "")
                #text = text.replace("  ", " ")
                text_ls.append((filename, text))
        elif re.search('.docx', filename) is not None:
            if re.search('.docx', filename) is not None:
                print(filename)
                text = docxpy.process(filename)
                text = text.strip()
                #text = text.replace("\n", "")
                #text = text.replace("\t", "")
                #text = text.replace("  ", " ")
                text_ls.append((filename, text))
        elif re.search('.csv', filename) is not None:
            with open(filename, "r", encoding="utf-8") as csv_file:
                print(filename)
                text= csv_file.read()
                #print(text)
                text_ls.append((filename, text))
        else: 
            pass


    """
    Replace the file path to the cleaned docs directory and replace .pdf or .docx with .txt.
    Save the file to that location.
    """
    for i in text_ls:
        if re.search('.pdf', i[0]) is not None:
            filepath = i[0].replace(doc_dir,clean_dir)
            filepath = filepath.replace(".pdf", ".txt")
            print(filepath)
        if re.search('.docx', i[0]) is not None:
            filepath = i[0].replace(doc_dir,clean_dir)
            filepath = filepath.replace(".docx", ".txt")
            print(filepath)
        if re.search('.csv', i[0]) is not None:
            filepath = i[0].replace(doc_dir,clean_dir)
            filepath = filepath.replace(".csv", ".txt")
            print(filepath)
        with open(filepath, 'w+', encoding="utf-8") as floc:
            floc.write(i[1])


if chunk_txt:
    ## Specify chunk size. Larger chunks means more context. Smaller is less.
    chunk_size = 1000
    chunk_ls = []

    ## Loop to iterate over all documents in directory and break them into n-size chunks
    for file in os.listdir(clean_dir):
        filename = os.path.join(clean_dir, file)
        text = open(filename, 'r', errors='ignore').read()
        chunks = split_into_chunks(text, chunk_size)
        for i in chunks:
            chunk_ls.append(i)


"""
Specify number of questions by size of random sample of chunks
"""

sample_size = 3

ls_rand = random.sample(chunk_ls, sample_size)

"""
Generate questions from chunk of text
"""

## Prompt for generating a question from a chunk of text
qa_gen_template = """
You will be generating questions based on content. Use the following content (delimited by <ctx></ctx>) and only the following content to formulate a question:
-----
<ctx>
{content}
</ctx>
-----
Answer:
)
"""

df_start = pd.DataFrame()

df = call_model(df_start, qa_gen_template, ['content'], ls_rand, OPENAI_API, gen_q = True)

"""
Generate answer from chunk and question
"""

## Prompt for generating a answer from a question and chunk of text
qa_answer_template = """
You will be answering questions based on content. Use the following content (delimited by <ctx></ctx>) and the question (delimited by <que></que>) to formulate an answer:
-----
<ctx>
{content}
</ctx>
-----
<que>
{question}
</que>
-----
Answer:
)
"""

df_end = call_model(df, qa_answer_template, ['content','question'], ls_rand, OPENAI_API, gen_q = False)

# """
# Save Model Outputs
# """

df_end.to_csv(os.path.join(out_dir, 'qa_output.csv'))