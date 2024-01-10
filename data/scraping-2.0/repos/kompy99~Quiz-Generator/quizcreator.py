# %pip install "evadb[document, notebook]"
# %pip install openai
# %pip install --upgrade tiktoken
# %pip install transformers

import argparse
import os
import evadb
import openai
import numpy as np
import random
import time
import tiktoken
import json
from transformers import BartTokenizer, BartForConditionalGeneration
from timeit import default_timer as timer

#Enter your OpenAI API key here
openai.api_key = "<OPENAI-API-KEY>"

# Enter the path to your PDF here
pdf_path = '<PDF-PATH>'

path = os.path.dirname(evadb.__file__)

def loadPDF(filepath, pdf_table_name, embeddings_table_name):
  cursor = evadb.connect(path).cursor()
  drop_pdf_table = f""" DROP TABLE IF EXISTS {pdf_table_name};"""
  load_pdf_data = f"""LOAD PDF '{filepath}' INTO {pdf_table_name};"""
  create_embedding_function = f"""CREATE FUNCTION IF NOT EXISTS get_embedding IMPL  '{path}/functions/sentence_feature_extractor.py'; """
  drop_embeddings_table = f""" DROP TABLE IF EXISTS {embeddings_table_name};"""
  get_pdf_embeddings = f"""CREATE TABLE IF NOT EXISTS {embeddings_table_name} AS SELECT get_embedding(data), data FROM {pdf_table_name};"""
  drop_embeddings_index = f""" DROP INDEX IF EXISTS embedding_index;"""
  build_faiss_index = f""" CREATE INDEX embedding_index ON {embeddings_table_name}(features) USING FAISS;"""

  cursor.query(drop_pdf_table).execute()
  cursor.query(load_pdf_data).execute()
  cursor.query(create_embedding_function).execute()
  cursor.query(drop_embeddings_table).execute()
  cursor.query(get_pdf_embeddings).execute()
  cursor.query(drop_embeddings_index).execute()
  cursor.query(build_faiss_index).execute()

def getPageCount(pdf_table_name: str) -> int:
  cursor = evadb.connect(path).cursor()
  get_page_count = f"""SELECT MAX(page) FROM {pdf_table_name} """
  page_counts_df = cursor.query(get_page_count).df()
  page_count = np.max(page_counts_df.loc[:, 'MAX.page'])
  return page_count

def getParagraphCount(pdf_table_name: str, page_number: int) -> int:
  cursor = evadb.connect(path).cursor()
  get_para_count = f"""SELECT page, MAX(paragraph) FROM {pdf_table_name} where page = {page_number}"""
  para_counts_df = cursor.query(get_para_count).df()
  para_count = np.max(para_counts_df.loc[:, 'MAX.paragraph'])
  return para_count

def generatePageSummary(pdf_table_name: str, page_number: int) -> str:
  cursor = evadb.connect(path).cursor()
  tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
  model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
  results = cursor.query(f"""SELECT page, paragraph, data from {pdf_data_table} where page = {page_number}""").df()
  dataKey = f'''{pdf_data_table}.data'''
  context = "\n".join(results[dataKey])
  tokenized_context = tokenizer.encode(context,truncation=True, return_tensors="pt")
  outputs = model.generate(tokenized_context, max_length=150, min_length=100, num_beams=4, length_penalty=2.0, early_stopping=True)
  generated_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
  return generated_summary

pdf_embeddings_table = "pdf_embeddings"
pdf_data_table = "pdf_table"

start_load_pdf = timer()
loadPDF(pdf_path, pdf_data_table, pdf_embeddings_table)
end_load_pdf = timer()

pdf_load_time = end_load_pdf - start_load_pdf

random.seed(time.time())
page_set = set()
num_pages = getPageCount(pdf_data_table)

for _ in range(5):
  random_page_number = random.randint(1, num_pages)
  page_set.add(random_page_number)

summaries = []
summary_generation_start_time = timer()
for page in page_set:
  generated_summary = generatePageSummary(pdf_table_name=pdf_data_table, page_number = page)
  print("\n Summary for page - " + str(page) + "\n---------------------\n")
  print("\n", generated_summary, "\n")
  summaries.append(generated_summary)

summary_generation_end_time = timer()

summary_generation_time =  summary_generation_end_time - summary_generation_start_time

summarized_context = "\n".join(summaries)

gpt_api_response_start_time = timer()
gptResponse = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": """You are a question generator. User will give content. You need to create multiple choice questions. Questions and answers have to be from the content only. Generate conceptual questions. Do not use any extra knowledge you may have on the subject.
        Every option must have a number assoicated with it. The answer must be the correct option number only. Generate 5 questions. Ensure that your output is in the following JSON format only. Sample question has been provided below:

        questions : [
          {
            question: What is 1+1?,
            options : [(1) 3, (2) 4, (3) 5, (4) 2],
            answer: 4
          }
        ]
        """},
        {"role": "user", "content": f"""{summarized_context}"""},
    ]
)

gpt_api_response_end_time = timer()
gpt_api_response_time = gpt_api_response_end_time - gpt_api_response_start_time

encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
num_tokens = len(encoding.encode(summarized_context))

print("\nMETRICS\n-------")
print("\nToken Usage\n-----------\nSummary tokens - " + str(num_tokens) + "\t Summary + Prompt tokens - " + str(gptResponse.usage.prompt_tokens) + "\t\tCompletion tokens - " + str(gptResponse.usage.completion_tokens))
print("\nTotal Tokens - " + str(gptResponse.usage.total_tokens))

print("\n\nPerformance\n-----------\nPdf loading - " + str(pdf_load_time) + " seconds \tSummary generation - " + str(summary_generation_time) + " seconds \tGPT API response - " + str(gpt_api_response_time) + " seconds")


quiz_data = json.loads(gptResponse.choices[0].message.content)
score = 0
num_questions = len(quiz_data['questions'])


print("\nYour practice quiz is ready!")

print("\nPRACTICE QUIZ\n--------------\n\n")
print(f"Instructions\n-------------\n\nThere will be {num_questions} questions in total. \nFor each question, enter only your choice (1,2,3 or 4). \nYou will see your score at the end.\n\nGood luck!!")

question_num = 0

print("\n\nQuiz\n------\n\n")
for question in quiz_data['questions']:
    question_num+=1
    print("Q" + str(question_num) + ") " + question['question'])
    for option in question['options']:
        print(option)
    user_answer = int(input("Your answer: "))
    if user_answer == question['answer']:
        print("Correct!\n")
        score+=1
    else:
        print(f"Sorry, the correct answer is: {question['answer']}\n")

print(f"\n\nYour score: {score}/{num_questions}")
