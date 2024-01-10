from dotenv import load_dotenv
from random import choice
import openai
import os
from flask import Flask, request
import ast


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY1")
completion = openai.Completion()



def str_to_dict(string):
    # remove the curly braces from the string
    string = string.strip('{}')
 
    # split the string into key-value pairs
    pairs = string.split(', ')

    tuple_pairs = [string.split(': ') for string in pairs]

    dic = {str(pair[0][1:-1]):float(pair[1]) for pair in tuple_pairs}
 
    # use a dictionary comprehension to create the dictionary, converting the values to integers and removing the quotes from the keys
    return dic



def emotion_responser(prompt_text):
  response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt_text,
        temperature=0.9,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=["\n\n\n"]
    )
  story = response['choices'][0]['message']['content']
  return str(story)





# def emotion_detector(text):
    
  '''
  I understood that while loop is not a good aproach for this as sometimes my model doesn't generate dictionary replys
  in this case, it goes through an infinite loop which leads to reaching limit of maximum request per minute for gpt
  '''
#     while True:
#       emotion_detection_prompt= [{'role':'system', 'content':"you are a emotion detection service that take a text as an input and return the emotions in the text. if there is no emotion in the text you must return just a python dictionary like: {'No emotion detected':0} . But if there are emotions in the text You return the name of the emotions without any explanation. the output template is a Python dictionary of 5 most probable emotions with their probability like:/n {emotion1:probability1, emotion2:probability2, emotion3:probability3, emotion4:probability4, emotion5:probability5} "}, {"role": "user", "content": f'input : {text}'}]

#       # emotion_detection_prompt= [{'role':'system', 'content':"you are a emotion detection service that take a text as an input\
#       #                              and return the emotions in the text. you just return the name of the emotions without any explanation.\
#       #                              if there is no emotion return 'no emotion detected'. the output template is a Python dictionary of\
#       #                              5 most probable emotions with their probability like:/n{'Neutral': 0.75, 'Sadness': 0.25, 'Fear': 0, 'Joy': 0, 'Anger': 0}\n\
#       #                             {'Neutral': 0.88, 'Surprise': 0.07, 'Sadness': 0.03, 'Anger': 0.02, 'Fear': 0.0}\n\
#       #                             {'anger': 0.6, 'disappointment': 0.3, 'neutral': 0.1, 'fear': 0.3, 'joy': 0.0}  "}, \
#       #                               {"role": "user", "content": text}]
#       story = emotion_responser(emotion_detection_prompt)
#       if story[0] == '{':
#          break
#       else:
#          continue
      
#     story = ast.literal_eval(story)
      
         
#     return str(story)
         
    
  
  

# # first simple function
"""
It works properly considering its promps and not having while loop. HOwever, I should develop it
"""
def emotion_detector(text):
    emotion_detection_prompt= [{'role':'system', 'content':"you are a emotion detection service that take a text as an input and return the emotions in the text. if there is no emotion in the text you must return just a python dictionary like: {'No emotion detected':0} . But if there are emotions in the text You return the name of the emotions without any explanation. the output template is a Python dictionary of 5 most probable emotions with their probability like:/n {emotion1:probability1, emotion2:probability2, emotion3:probability3, emotion4:probability4, emotion5:probability5} "}, {"role": "user", "content": f'input : {text}'}]
    story = emotion_responser(emotion_detection_prompt)
      # emotion_dic = str_to_dict(story)
    
    return story