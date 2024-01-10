import openai
import time
from provotype.prep import prepare_json_topics
import json
from time import sleep
import logging



model_id = 'gpt-3.5-turbo'

def generate_summarizer(my_tokens,prompt):

    
    prompts = 0
    conversation=[]

    while conversation==[]:


        if (prompts % 3) == 0 and prompts !=0:
                print("waiting for 60 seconds")
                sleep(60) 

        try:

            res = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                max_tokens=my_tokens,
                temperature=0.7,
                top_p=0.5,
                frequency_penalty=0.5,
                messages=
               [
                 {
                  "role": "system",
                  "content": "You are a helpful assistant for text summarization.",
                 },
                 {
                  "role": "user",
                  "content": f"Can you make a summariziation for following text: {prompt}",
                 },
                ],
            )
            conversation = res["choices"][0]["message"]["content"]
        except Exception as e:
            print(f'during the summarization following exception occured:{str(e)}')
            prompts=prompts+1


    return conversation 



def do_summarization(split_text,number_splits,response_max_tokens):

    from time import sleep

    summarized_text = []

    nmb_splits = number_splits
    
    print(number_splits)
   
    max_tokens = response_max_tokens


    for i in range(nmb_splits):

        if (i % 3) == 0 and i !=0:
            print("waiting for 60 seconds")
            sleep(60)
  
        summ_text = generate_summarizer(max_tokens,split_text[i])
        summarized_text.append(summ_text)
    
      
    return summarized_text




def create_five_topics(text_data):

    prompts = 0
    
    json_str =  """{"topics": {"topic:","","rating:",""}}"""

    conversation = []

    while conversation == []:

        if (prompts % 3) == 0 and prompts !=0:
            print("waiting for 60 seconds")
            sleep(60)


        try:
    
            response = openai.ChatCompletion.create(
                model = model_id,
                messages = [{'role':'system', 'content': 'You are a helpful research assistant.'},
                            {'role': 'user', 'content':f"Give me the five most relevant topics plus a probability between 0 and 1. \
                            Use only one word for each topic and sort from the highest til the lowest value.\
                            Return it in a JSON object like {json_str} for the following text: {text_data}"},
                           ])
            
            api_usage = response['usage']
            print('Total token consumed: {0}'.format(api_usage['total_tokens']))
          
            conversation.append({'role': response.choices[0].message.role, 'content': response.choices[0].message.content})
            

        except Exception as e: 
            print(f'during the topics following exception occured:{str(e)}')
            prompts = prompts+1



        try:
           
            sorted_dict_topis = prepare_json_topics(conversation)

        except:
            conversation = []
    
    
    return sorted_dict_topis


def summarize_summarized_texts(summarized_text):
    conversation = []

    prompts=0

    while conversation==[]:


        if (prompts % 3) == 0 and prompts !=0:
                print("waiting for 60 seconds")
                sleep(60) 

        try:


      
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                max_tokens = 100,
             
                temperature=0.7,
                top_p=0.5,
                frequency_penalty=0.5,
                messages=
               [
                 {
                  "role": "system",
                  "content": "You are a helpful assistant for text summarization.",
                 },
                 {
                  "role": "user",
                  "content": f"Can you make a summariziation with the maximum number of tokens for following text: {summarized_text}",
                 },
                ],
            )
            
            conversation.append({'role': response.choices[0].message.role, 'content': response.choices[0].message.content})
        except Exception as e:
            print(f'during the summarization of all summaries following exception occured:{str(e)}')    
            promts = prompts +1
        
    return conversation



def scale_conversation(text_data):
    prompts = 0
    json_str =  """{"scales": {"scale:","","rating:",""}}"""
    conversation = []

    while conversation == []:

        if (prompts % 3) == 0 and prompts !=0:
            print("waiting for 60 seconds")
            sleep(60)


        try:

            print('try to get response scale')
            response = openai.ChatCompletion.create(
                model = model_id,
                messages = [{'role':'system', 'content': 'You are a helpful research assistant.'},
                            {'role': 'user', 'content':f"return how emotional between 0 and 10, controversial between 0 and 10, factual between 0 and 10, sensitive\
                        between 0 and 10 in a JSON object like {json_str} is following text (treat the individual strings as complete text): {text_data}"},
                           ])
            
            api_usage = response['usage']
    
  
            conversation.append({'role': response.choices[0].message.role, 'content': response.choices[0].message.content})

        except: 
            pass
            print('response scale did not work')
            prompts = prompts+1



        try:
            print('try scale json')

            list_scale, list_rating = prepare_json_scale(conversation)

        except:
            print('json did not work')
            conversation = []
    

       




def write_a_haiku(summarized_text):
    conversation = []
    prompts=0

    while conversation == []:

        if (prompts % 3) == 0 and prompts !=0:
            print("waiting for 60 seconds")
            sleep(60)


        try:

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                max_tokens = 100,
             
                temperature=0.7,
                top_p=0.5,
                frequency_penalty=0.5,
                messages=
               [ {
                  "role": "system",
                  "content": "You are a creative writer.",
                 },
                
                 {
                  "role": "user",
                  "content": f"Can you write me a haiku for following text: {summarized_text}",
                 },
                ],
            )
            
            conversation.append({'role': response.choices[0].message.role, 'content': response.choices[0].message.content})
        except Exception as e:
            print(f'during haiku following exception occured:{str(e)}') 
            prompts = prompts +1
    
    return conversation


def create_image(text):

    image_url = []

    prompts = 0

    while image_url==[]:

        if (prompts % 3) == 0 and prompts !=0:
            print("waiting for 60 seconds")
            sleep(60)


        try:

            user_prompt = text
            response = openai.Image.create(
                prompt = user_prompt,
                n=1,
                size = "512x512"
            )
            image_url = response['data'][0]['url']


        except Exception as e:
            print(f'during image following exception occured:{str(e)}') 
            prompts = prompts +1

    return image_url
