# top part is template code

# make sure to 
# pip install cohere altair
# pip install scikit-learn

import requests
import cohere
import random
import pandas as pd
import numpy as np
import sklearn

api_key = 'L2VMOXwleskZQjVuP5QEe2puJTKNLAzGaRhSEVTK'
co = cohere.Client(api_key)

def generate_text(prompt, temp=0):
  response = co.generate(
    model='command',
    prompt=prompt,
    max_tokens=200,
    temperature=temp)
  return response.generations[0].text

# Questions and answers are here for testing purposes.
# In the question-gen.py file, the q_s and a_s arrays are generated from the user input and should be used in place here.
# These are just arrays from one iteration of running question-gen.py
q_s = ['Question: What is the main idea of this lesson?', 'Question: Why is it expensive to connect and disconnect from a database multiple times?', 'Question: What is the Singleton Pattern?', 'Question: Why is it beneficial for the environment to save energy and processing power?']
a_s = ['Answer: Connecting to databases can be expensive in terms of time and resources, therefore, it is efficient to connect once and perform all the necessary functions dealing with the database.', 'Answer: Because it requires spending time and company resources for each connection and disconnection.', 'Answer: The Singleton Pattern is a design pattern that restricts the instantiation of a class to one object. It is often used for database connections.', 'Answer: Because the excessive connections and disconnections to the database are avoided. This, in turn, reduces the carbon footprint.']

# This part evaluates whether user answers are correct or incorrect

counter = len(q_s)
for i in range(counter):
  if q_s[i] and a_s[i]:
    print(q_s[i])
    usr_ans = input("Answer TutorBo! ")
    real_ans = (a_s[i])
    # embedding for user answer: sentence1
    sentence1 = np.array(co.embed([usr_ans]).embeddings)
    # embedding for tutor answer: sentence2
    sentence2 = np.array(co.embed([real_ans]).embeddings)

    from sklearn.metrics.pairwise import cosine_similarity
    # cosine similarity between sentence 1 and sentence 2
    rating = cosine_similarity(sentence1, sentence2)[0][0]
    if rating >= 1.3 or rating <= 0.7:
      # Incorrect Answer
      print(bool(False))
    else:
      # Correct Answer
      print(bool(True))
      
