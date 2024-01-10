from dotenv import load_dotenv
import os
import openai
import time
import numpy as np

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,plot_confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

good_reviews = []
bad_reviews = []
for i in range(0,500):
  completion_good = openai.Completion.create(engine="text-davinci-003", 
                                            prompt="This hotel was great.",
                                            max_tokens=512)
  good_reviews.append(completion_good.choices[0]['text'])
  completion_bad = openai.Completion.create(engine="text-davinci-003", 
                                            prompt="This hotel was terrible.",
                                            max_tokens=512)
  bad_reviews.append(completion_bad.choices[0]['text'])
  display = np.random.choice([0,1],p=[0.7,0.3])
  time.sleep(3)
  if display ==1:
    display_good = np.random.choice([0,1],p=[0.5,0.5])
    if display_good ==1:
      print('Printing random good review')
      print(good_reviews[-1])
    if display_good ==0:
      print('Printing random bad review')
      print(bad_reviews[-1])
      
import pandas as pd
import numpy as np
df = pd.DataFrame(np.zeros((944,2)))
df.columns = ['Reviews','Sentiment']
df['Sentiment'].loc[0:499] = 1
df['Reviews'] = good_reviews+bad_reviews
df.to_csv('Classification/Datasets/generated_reviews.csv')

labeled_data = pd.read_csv('Classification/Datasets/generated_reviews.csv'.drop(columns=['Unnamed: 0']))
labeled_data.Sentiment = labeled_data.Sentiment.astype(int)
labeled_data = labeled_data.dropna().reset_index()

dataset = labeled_data
from transformers import AutoTokenizer

#tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#tokenized_data = tokenizer(dataset["Reviews"].values.tolist(), return_tensors="np", padding=True)
vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8)
tokenized_data = vectorizer.fit_transform(dataset['Reviews']).toarray()

labels = np.array(dataset["Sentiment"])  # Label is already an array of 0 and 1
rf = RandomForestClassifier(n_estimators=100)
X = tokenized_data
y = labels
X_train, X_test,y_train, y_test = train_test_split(X,y,test_size=0.2)
rf.fit(X_train,y_train)
plot_confusion_matrix(rf,X_test,y_test)


import openai

api_key = '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'

# Define prompt and model
prompt = 'Extract the snippet of text containing the end <date> , lease <price>, designated <use>, and leased space <area> from the following statement in a legal contract: 6.1 Both parties agree that, the lease term commences from July 1, 2012 and ends on June 30, 2015. The unit rent per square meter of construction area of the Premises per day is adjusted to RMB Four Point Two Zero (RMB4.20), adding up to an annual rental of RMB Eight Million Two Hundred Seventy Thousand Seven Hundred Forty Nine Point Six Two (RMB8,270,749.62), and agreed that the space can only be used for meetings only; \n\n-->\n\n'
model = "text-davinci-003"

# Make API request
import openai
openai.api_key = api_key
response = openai.Completion.create(
  engine=model,
  prompt=prompt,
  max_tokens=500,
  n=1,
  stop=None,
  temperature=0.5,
)

# Print completion
print(response["choices"][0]["text"])




