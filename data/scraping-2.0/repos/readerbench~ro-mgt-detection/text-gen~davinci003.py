'''
Text Generation - Text Completion + Backtranslation (Ro-Ru-Tr-Ro)
Run OpenAI text-davinci-003 for text completion and backtranslation tasks.

Disclaimer: Make sure you have an OpenAI account or an Azure Subscription to be able to run the model via API. 
The API_KEY is required.
'''

import pandas as pd
import os
import openai

os.environ['OPENAI_API_KEY'] = 'insert-your-API_KEY'
openai.api_type = "azure"
openai.api_base = "insert-your-endpoint"
openai.api_version = "2022-12-01"
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load your input dataset (human text)
df=pd.read_pickle('insert-dataset-path')

###### Text Completion Task #########

# Take the first 10 tokens from each human text to be given as input to the text generation model
df["AI_Input"] = df["HumanText"].str.split().str[:10].str.join(sep=" ")
df["AI_Input"] = df["AI_Input"].astype('string')
df = df.dropna()
mylist = df.AI_Input.to_list()
mylist = list(dict.fromkeys(mylist))

completed_texts = []
for start_phrase in mylist:
  response = openai.Completion.create(
  engine="davinci-003",
  prompt = start_phrase,
  temperature=1,
  max_tokens=512,
  top_p=0.5,
  frequency_penalty=0,
  presence_penalty=0,
  best_of=1,
  stop=None)

  completed_text = response['choices'][0]['text'].replace('\n', '').replace(' .', '.').strip()
  completed_texts.append(start_phrase+" "+completed_text)

result_df = pd.DataFrame({'input': mylist, 'output': completed_texts})

# Save it in format text - label- description
output_frame= result_df[["output"]]
output_frame['label'] = '0'
output_frame['description'] = 'This is AI-generated text'
output_frame.rename(columns = {'Output':'text'}, inplace = True)
output_frame.to_csv('ai-text-davinci003-completion.csv',index = False, encoding = 'utf-8-sig')


###### Backtranslation Task #########

df['text'] = df["text"].astype('string')
df = df.dropna()
mylist2 = df.text.to_list()
mylist2 = list(dict.fromkeys(mylist))

# Ro -> Ru
translated_texts = []
for start_phrase in mylist2:
    #max_prompt_length = 512
    truncated_prompt = start_phrase#[:max_prompt_length]
    response = openai.Completion.create(
    engine="davinci-003",
    prompt = f"Translate the following from Romanian to Russian.\n{truncated_prompt}",
    temperature=0.7,
    max_tokens=500,
    #top_p=0.5,
    frequency_penalty=0,
    presence_penalty=0,
    best_of=1,
    stop=None,
    timeout=600)

    #translated_text = response['choices'][0]['text'].replace('\n', '').replace(' .', '.').strip()
    translated_text = ''.join([choice['text'] for choice in response['choices']]).replace('\n', '').replace(' .', '').strip()

    translated_texts.append(translated_text)

translated_df1 = pd.DataFrame({'input': mylist2, 'output': translated_texts})

# Ru -> Tr
new_df = translated_df1['output']
test_list1 = new_df.to_list()
test_list1 = list(dict.fromkeys(test_list1))

translated_texts_2 = []
for start_phrase in test_list1:
  #max_prompt_length = 512
  truncated_prompt = start_phrase#[:max_prompt_length]
  response2 = openai.Completion.create(
  engine="davinci-003",
  prompt = f"Translate the following from Russian to Turkish.\n{truncated_prompt}",
  temperature=0.7,
  max_tokens=500,
  #top_p=0.5,
  frequency_penalty=0,
  presence_penalty=0,
  best_of=1,
  stop=None,
  timeout=600)

  #translated_text_2 = response2['choices'][0]['text'].replace('\n', '').replace(' .', '.').strip()
  translated_text_2 = ''.join([choice['text'] for choice in response2['choices']]).replace('\n', '').replace(' .', '').strip()

  translated_texts_2.append(translated_text_2)

translated_df2 = pd.DataFrame({'input': test_list1, 'output': translated_texts_2})

#Tr -> Ro
new_df_2 = translated_df2['output']
test_list2 = new_df_2.to_list()
test_list2 = list(dict.fromkeys(test_list2))

translated_texts_3 = []
for start_phrase in test_list2:
  #max_prompt_length = 500
  truncated_prompt = start_phrase#[:max_prompt_length]
  response3 = openai.Completion.create(
  engine="davinci-003",
  prompt = f"Translate the following from Turkish to Romanian.\n{truncated_prompt}",
  temperature=0.7,
  max_tokens=500,
  #top_p=0.5,
  frequency_penalty=0,
  presence_penalty=0,
  best_of=1,
  stop=None,
  timeout=600)

  #translated_text_3 = response3['choices'][0]['text'].replace('\n', '').replace(' .', '.').strip()
  translated_text_3 = ''.join([choice['text'] for choice in response3['choices']]).replace('\n', '').replace(' .', '').strip()

  translated_texts_3.append(translated_text_3)

df_final = pd.DataFrame({'input': mylist2, 'output': translated_texts_3})
df_final.to_csv('ai-text-davinci003-backtranslated.csv',index = False, encoding = 'utf-8-sig')
