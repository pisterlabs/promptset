'''
Description: this is a program of analyzing the data collected from twitter.  openAI API is used.

Authors: Zequan Tang

Date: June 25, 2023
'''
import csv
import openai
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6)) # use Tenacity library to solve API rate limit
def callAPI(text):
  return openai.Completion.create(
        engine = 'text-davinci-003',
        prompt = (f"Analyze the attitude of the following text to the topic of whether AI will replace human jobs in the future. Your answer should select from 'will replace', 'won't replace' and 'neutral'. Text: '{text}'"),
        max_tokens = 10,
        temperature = 0
      )

positive = 0
negative = 0
neutral = 0
total = 0
data = []
openai.api_key = 'YOUR OWN API KEY'

with open("./tweets_Job.csv", 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    # get the content
    content = row[2]            
    if(str(content) == 'Content'):
      continue
    text = str(content.replace('\n', ' '))

    # call openAI api
    completion = callAPI(text)

    if ("won't replace" in completion.choices[0].text.strip().lower()):
      positive += 1
      tuple1 = (text, 'Will not replace')
      data.append(tuple1)
    
    if ("will not replace" in completion.choices[0].text.strip().lower()):
      positive += 1
      tuple2 = (text, 'Will not replace')
      data.append(tuple2)

    if ("neutral" in completion.choices[0].text.strip().lower()):
      neutral += 1
      tuple3 = (text, 'Neutral')
      data.append(tuple3)

    if ("will replace" in completion.choices[0].text.strip().lower()):
      negative += 1
      tuple4 = (text, 'Will replace')
      data.append(tuple4)

    '''
    # print out the count of each category
    print(positive, '\n')
    print(neutral, '\n')
    print(negative, '\n')
    '''  
    
# write back to a csv file
fileName = 'dataOpenAI.csv'
with open(fileName, 'w', newline='') as csvfile:
        header = ['Content', 'Result']
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(data)

# data visualization
categories = ['Will not replace', 'Neutral', 'Will replace']
data = [positive, neutral, negative]
colors = ['#458B74', '#7EC0EE', '#A6A6A6']

def func(pct, data):
    absolute = int(pct / 100.*np.sum(data))
    return "{:.1f}%\n({:d})".format(pct, absolute)

plt.pie(data, explode=None, labels=categories, colors=colors, startangle=90,radius = 1.2, autopct=lambda pct: func(pct, data), shadow=True)
plt.title("Distribution of 5,000 Twitter users' opinions on whether AI will replace human jobs in the future (by openAI)")
plt.legend()
plt.show()