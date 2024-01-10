import os
import openai
import json
import pandas as pd
import ast

openai.api_key = os.environ['OPENAI_KEY']
df = pd.read_csv('transcript_sample.csv')


def concat_string(df: pd.DataFrame, column_name: str) -> str:
  # Replace NaN values with '[]'
  df[column_name] = df[column_name].fillna('[]')
  # Convert strings to lists
  df[column_name] = df[column_name].apply(ast.literal_eval)

  # Extract string from list in each row
  df[column_name] = df[column_name].apply(lambda x: x[0] if len(x) > 0 else '')

  # Concatenate strings into a single list
  output_list = df[column_name].tolist()
  output_string = ", ".join(output_list)

  return output_string


DEFAULT_MODEL = 'gpt-3.5-turbo'


def get_top_5(text):
  message = f"""The following text includes \
  top concerns from 60 calls for a puppy company.\
  help me summarize what are the top 5 topics and \
  what % of total it makes up. If there are more \
  than one topic in each sentence, break them \
  into two. Merge topics if they are similar \
  and relevant. 
Text:
---
{text}
---
Output as one json object with Topic and Percentage as keys\

```json
"""
  response = openai.ChatCompletion.create(
    model=DEFAULT_MODEL,
    messages=[{
      "role": "user",
      "content": message
    }],
    stop=None,
    temperature=0,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
  )
  json_result = response.choices[0].message.content
  result = json.loads(json_result)
  return result


print("getting top 5 topics...")
top_5_concerns = get_top_5(concat_string(df, 'Concerns'))
print(top_5_concerns)
df_concerns = pd.DataFrame(top_5_concerns)
print(df_concerns.columns)
top_5_feedback = get_top_5(concat_string(df, 'Improvements'))
print(top_5_feedback)
df_feedback = pd.DataFrame(top_5_feedback)
print(df_feedback.columns)


def get_relevant_quotes(topic, text):
  message = f"""The following text includes \
  example questions from 60 sales call for a puppy \
  company. output three quotes that are highly related to '{topic}'\
Text:
---
{text}
---
Output the 3 quotes as a list 

"""
  response = openai.ChatCompletion.create(
    model=DEFAULT_MODEL,
    messages=[{
      "role": "user",
      "content": message
    }],
    stop=None,
    temperature=0,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
  )
  result = response.choices[0].message.content
  return result


print("getting relevant quote for each topic...")
concern_quotes = concat_string(df, 'Quotes')
improvement_quotes = concat_string(df, 'Improvement_Quotes')
df_concerns['quotes'] = df_concerns['Topic'].apply(
  lambda x: get_relevant_quotes(concern_quotes, x))
df_concerns.to_csv('concern_output.csv', index=False)
df_feedback['quotes'] = df_feedback['Topic'].apply(
  lambda x: get_relevant_quotes(improvement_quotes, x))
df_feedback.to_csv('feedback_output.csv', index=False)
