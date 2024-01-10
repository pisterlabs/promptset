import openai
import time
from provotype.prep import prepare_json_topics
import json
from time import sleep
import logging



model_id = 'gpt-3.5-turbo'

def generate_summarizer_de(my_tokens,prompt):

    
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
                  "content": "Du bist ein hilfreicher Assistent für Textzusammenfassungen.",
                 },
                 {
                  "role": "user",
                  "content": f"Kannst du mir eine Zusammenfassung für folgenden Text erstellen: {prompt}",
                 },
                ],
            )
            conversation = res["choices"][0]["message"]["content"]
        except Exception as e:
            print(f'during the summarization following exception occured:{str(e)}')
            prompts=prompts+1


    return conversation 



def do_summarization_de(split_text,number_splits,response_max_tokens):

    from time import sleep

    summarized_text = []

    nmb_splits = number_splits
    
    print(number_splits)
   
    max_tokens = response_max_tokens


    for i in range(nmb_splits):

        if (i % 3) == 0 and i !=0:
            print("waiting for 60 seconds")
            sleep(60)
  
        summ_text = generate_summarizer_de(max_tokens,split_text[i])
        summarized_text.append(summ_text)
    
      
    return summarized_text




def create_five_topics_de(text_data):

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
                messages = [{'role':'system', 'content': 'Du bist ein hilfreicher Forschungsassistent.'},
                            {'role': 'user', 'content':f"Gib mir die fünf relevantesten Themen plus eine Wahrscheinlichkeit zwischen 0 und 1 \
                             Verwende jeweils nur ein Wort für jedes Thema und sorte nach höchstem bis niedrigstem Wert\
                             Gib es in einem JSON-Objekt wie {json_str} zurück für den folgenden Text: {text_data}"},
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


def summarize_summarized_texts_de(summarized_text):
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
                  "content": "Du bist ein hilfreicher Assistent für Textzusammenfassungen.",
                 },
                 {
                  "role": "user",
                  "content": f"Kannst du mir eine Zusammenfassung mit der maximalen Anzahl von Tokens für folgenden Text erstellen: {summarized_text}",
                 },
                ],
            )
            
            conversation.append({'role': response.choices[0].message.role, 'content': response.choices[0].message.content})
        except Exception as e:
            print(f'during the summarization of all summaries following exception occured:{str(e)}')    
            promts = prompts +1
        
    return conversation
 




def write_a_haiku_de(summarized_text):
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
                  "content": "Du bist ein kreativer Dichter.",
                 },
                
                 {
                  "role": "user",
                  "content": f"Kannst du mir ein Haiku über folgenden Text schreiben: {summarized_text}",
                 },
                ],
            )
            
            conversation.append({'role': response.choices[0].message.role, 'content': response.choices[0].message.content})
        except Exception as e:
            print(f'during haiku following exception occured:{str(e)}') 
            prompts = prompts +1
    
    return conversation


