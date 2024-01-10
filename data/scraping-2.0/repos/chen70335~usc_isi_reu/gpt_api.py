# Uses the OpenAI GPT API to rephrase sentences and create annotation files for research purposes

import openai
import pandas as pd
import numpy as np
import random
# openai.api_key =

# Calls GPT API to rephrase sentences with less emotional and subjective tone
def call_gpt(text):
  text = text
  prompt = "Rewrite the following sentence to show less negative emotions and more neutral and objective tone: "
  message = prompt + text
  res = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
          {"role": "user", "content": message}
      ]
  )
  ret = res['choices'][0]['message']['content']
  print(ret)
  return ret

# Randomly sample sentences from a dataset
def random_sample(all_df):
  id1_1 = pd.read_csv('data/annot/id1_sentence_pairs_annotations_no_shuffling.csv')
  id1_1_ids = list(id1_1['id'])
  n = 0
  all_df_sample = pd.DataFrame()
  while n != 40:
    row = all_df.sample(n=1)
    sent = str(row['sentence']).encode('ascii', 'ignore').decode('ascii')
    print(row)
    print(row['id'].iloc[0], type(row['id'].iloc[0]))   
    print(row['id'].iloc[0] in id1_1_ids) 
    while row['id'].iloc[0] in id1_1_ids or '"' in sent or '@' in sent or '“' in sent or 'â€' in sent or '”' in sent: 
      print(sent)
      row = all_df.sample(n=1)
      sent = str(row['sentence']).encode('ascii', 'ignore').decode('ascii')
      print('Resample!')
    all_df_sample = pd.concat([all_df_sample, row])
    n += 1
  return all_df_sample

# Takes either a random sample from random_sample() or the max 30 scoring sentences from max_sents_annot()
# Outputs two annotation files of which one is randomly shuffled
def main(all_df_sample, filename):
  all_df_sample = all_df_sample.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
  # Move 'sentence' column to the end
  all_df_sample = all_df_sample[[c for c in all_df_sample if c not in ['sentence']] 
        + ['sentence']]
  rephrasing = []  
  for i, row in all_df_sample.iterrows():
    print(row['sentence'], type(row['sentence']))
    rephrasing.append(call_gpt(row['sentence']))
  all_df_sample['rephrasing'] = rephrasing
  all_df_sample.to_csv(filename + 'no_shuffling.csv')

  shuffled_sample = all_df_sample[['sentence', 'rephrasing']]
  # Shuffle every row selected columns randomly
  shuffled_sample = shuffled_sample.apply(lambda x: pd.Series(random.sample([x['sentence'], x['rephrasing']], k=2)), axis=1)
  shuffled_sample.to_csv(filename + 'shuffled.csv')

# Sort the dataframe by sentiment intensity and subjectivity and take the top 30 scoring sentences
def max_sents_annot(df):
	sorted_df = df.sort_values(by=['sentiment_intensity', 'subjectivity'], ascending=[True, False])
	max_30 = sorted_df.iloc[0:30]
	return max_30
  


if __name__ == "__main__":
  all_df = pd.read_csv('data/polusa_sentenced/all_news_reporting.csv', encoding='utf-8')
  # Example call to generate two annotation files for the 30 max scoring sentences and its GPT rephrasing sentences
  main(max_sents_annot(all_df), 'data/annot/id3.2_max_sentence_pairs_')
