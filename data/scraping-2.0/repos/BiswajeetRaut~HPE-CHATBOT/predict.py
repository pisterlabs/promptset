import pickle
import tensorflow as tf
import tflearn
import numpy as np
import random
import json
import nltk
import openai
openai.api_key = "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"
nltk.download("punkt")
from tensorflow import keras

from nltk.stem.lancaster import LancasterStemmer
stemmer=LancasterStemmer()
data = pickle.load( open( "training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

with open('intents.json') as json_data:
  intents = json.load(json_data)
# load the saved model
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Defining model and setting up tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
model.load("model.tflearn") 


# First whenever we get a sentence from the user, we need to clean it up
def clean_up_sentence(sentence):
  # Tokenizing the pattern
  sentence_words = nltk.word_tokenize(sentence)
  # Performing stemming on each word
  sentence_words=[stemmer.stem(word.lower()) for word in sentence_words]
  return sentence_words

# Then we need to define the bag of word which will act as an input to the model
def bow(sentence, words, show_details=False):
  # Tokenizing the sentence
  sentence_words=clean_up_sentence(sentence)
  # Generating the bag of words
  bag=[0]*len(words)
  for s in sentence_words:
    for i,w in enumerate(words):
      if w == s: 
        bag[i] = 1
        # if show_details:
          # print("found in bag: %s" % w)
  return (np.array(bag))

context = {}

ERROR_THRESHOLD = 0.25
def classify(sentence):
  # generate probabilities from the model
  results = model.predict([bow(sentence, words)])[0]
  # filter out predictions below a threshold
  # for i,r in enumerate(results):
  #   if r>ERROR_THRESHOLD:
  #      print(r)
  results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
  # sort by strength of probability
  results.sort(key=lambda x: x[1], reverse=True)
  return_list = []
  # print(results)
  for r in results:
    return_list.append((classes[r[0]], r[1]))
    
    # return tuple of intent and probability
  return return_list

def response(sentence, userID='123', show_details=False):
  question=sentence+""
  # print(question)
  results = classify(sentence)
  # print(results)
  # if we have a classification then find the matching intent tag
  if results:
    # print(question)
    # loop as long as there are matches to process
    while results:
      for i in intents['intents']:
        # find a tag matching the first result
        if i['tag'] == results[0][0]:
          # set context for this intent if necessary
          if 'context_set' in i:
            # if show_details: print ('context:', i['context_set'])
            context[userID] = i['context_set']

          # check if this intent is contextual and applies to this user's conversation
          if not 'context_filter' in i or \
            (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
            if show_details: print ('tag:', i['tag'])
            # a random response from the intent
            # print(question)
            # output = openai.ChatCompletion.create( model="gpt-3.5-turbo",messages=[ {"role": "user", "content":"share some insight on HPE Primera storage 5000" } ] )
            # print(output.choices[0].message.content) 
            return random.choice(i['responses'])
            # print(output.choices[0].message.content) 

      results.pop(0)
  else:
    output = openai.ChatCompletion.create( model="gpt-3.5-turbo",messages=[ {"role": "user", "content":question } ] )
    return output.choices[0].message.content
# response("hi")
# response("share some insight on HPE Primera storage 5000")
     