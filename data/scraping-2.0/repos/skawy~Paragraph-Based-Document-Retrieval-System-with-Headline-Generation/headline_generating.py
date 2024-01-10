import os
import pandas as pd
import openai
import time


#Use Your Api Key Here
os.environ["OPENAI_API_KEY"] = '******'

dataset = pd.read_csv('BBC News Train.csv')

sport_dataset=dataset.query("Category == 'sport'")
sport_dataset

sport_paragraphs = sport_dataset.Text

client = openai.Client()

# Iam Printing The First 3 Paragraphs just for testing
for paragraph in sport_paragraphs[:3]:
  response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {
        "role": "system",
        "content": "Craft a reflective Headline that highlights the broader significance of the upcoming paargraph."
      },
      {
        "role": "user",
        "content": paragraph
      }
    ],
    temperature=0.8,
    max_tokens=32,
    top_p=0.8
  )

  assistant_response = response.choices[0].message.content

  print(f'\n\n==============================================================\n\n')
  print(f'Paragraph: {paragraph}')
  print(f'\n\n')
  print(f'Headline: {assistant_response}')
  # Rate limit reached for gpt-3.5-turbo then iam delaying the request
  time.sleep(20)