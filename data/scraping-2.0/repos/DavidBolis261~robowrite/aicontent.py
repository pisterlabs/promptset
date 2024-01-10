
import openai
import config
import os
openai.api_key = config.OPENAI_API_KEY


def productDescription(query):
    openai.api_key = openai.api_key
    response = openai.Completion.create(
    model="davinci-instruct-beta-v3",
    prompt="generate a detailed product descriptions for: {}".format(query),
    temperature=0.7,
    max_tokens=300,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    if 'choices' in response:
        if len(response['choices']) > 0:
           answer = response['choices'][0]['text']
    
        else:
           answer = 'Sorry! I am not sure how to help you this time.'
    else:
           answer = 'Sorry! I am not sure how to help you this time.'
    return answer

def sendEmail(query):
    openai.api_key = openai.api_key
    response = openai.Completion.create(
    model="davinci-instruct-beta-v3",
    prompt="generate a cold email  for: {}".format(query),
    temperature=0.7,
    max_tokens=300,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    if 'choices' in response:
        if len(response['choices']) > 0:
           answer = response['choices'][0]['text']
    
        else:
           answer = 'Sorry! I am not sure how to help you this time.'
    else:
           answer = 'Sorry! I am not sure how to help you this time.'
    return answer

def cuteMessages(query):
    openai.api_key = openai.api_key
    response = openai.Completion.create(
    model="davinci-instruct-beta-v3",
    prompt="generate a long cute message  for: {}".format(query),
    temperature=0.7,
    max_tokens=300,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    if 'choices' in response:
        if len(response['choices']) > 0:
           answer = response['choices'][0]['text']
    
        else:
           answer = 'Sorry! I am not sure how to help you this time.'
    else:
           answer = 'Sorry! I am not sure how to help you this time.'
    
    return answer

def jobDescription(query):
    openai.api_key = openai.api_key
    response = openai.Completion.create(
    model="davinci-instruct-beta-v3",
    prompt="write a detailed job description for: {}".format(query),
    temperature=0.7,
    max_tokens=300,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    if 'choices' in response:
        if len(response['choices']) > 0:
           answer = response['choices'][0]['text']
    
        else:
           answer = 'Sorry! I am not sure how to help you this time.'
    else:
           answer = 'Sorry! I am not sure how to help you this time.'
    
    return answer

def socialmediaads(query):
    openai.api_key = openai.api_key
    response = openai.Completion.create(
    model="davinci-instruct-beta-v3",
    prompt="write a social media advertisement for: {}".format(query),
    temperature=0.7,
    max_tokens=300,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    if 'choices' in response:
        if len(response['choices']) > 0:
           answer = response['choices'][0]['text']
    
        else:
           answer = 'Sorry! I am not sure how to help you this time.'
    else:
           answer = 'Sorry! I am not sure how to help you this time.'
    
    return answer

def coverLetter(query):
      openai.api_key = openai.api_key
      response = openai.Completion.create(
      model="davinci-instruct-beta-v3",
      prompt="write a detailed cover letter for: {}".format(query),
      temperature=0.7,
      max_tokens=300,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
      )
      if 'choices' in response:
         if len(response['choices']) > 0:
            answer = response['choices'][0]['text']
      
         else:
            answer = 'Sorry! I am not sure how to help you this time.'
      else:
            answer = 'Sorry! I am not sure how to help you this time.'
      
      return answer

def youtubeVideoIdea(query):
      openai.api_key = openai.api_key
      response = openai.Completion.create(
      model="davinci-instruct-beta-v3",
      prompt="list YouTube video ideas for: {}".format(query),
      temperature=0.7,
      max_tokens=300,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
      )
      if 'choices' in response:
         if len(response['choices']) > 0:
            answer = response['choices'][0]['text']
      
         else:
            answer = 'Sorry! I am not sure how to help you this time.'
      else:
            answer = 'Sorry! I am not sure how to help you this time.'
      
      return answer