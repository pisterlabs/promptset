import openai
import time
import re

openai.api_key = ''



def main_chat(entity):       #use this function to generate wikipedia
  sys_prompt = "Answer the following question only if you know the answer or can make a well-informed guess; otherwise tell me you don't know it"
  instructs = "Please write a brief Wikipedia for {} under 100 words."    
  prompt = instructs.format(entity)
  
  response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt}],
      max_tokens=256,
      temperature=0           
      )
  text_response = response["choices"][0]["message"]["content"]
  return text_response


def question_generation(entity,content):
  instructs = "I will give you some information about the entity. You should use all this information to generate a question, and the answer to your question is the entity. Do not include the entity in your question.\
  \nThere is an example.\nentity:World War II\ninformation: World War II, also known as the Second World War, was a global war that lasted from 1939 to 1945.\nquestion: which global war lasted from 1939 to 1945?\n\
  entity: {}\ninformation: {}\nquestion:"     
  prompt = instructs.format(entity,content)
  response = request_api(prompt)
  return response
      
def reverse_modeling(question):
  prompt = "You should answer the following question as short as possible. {}"     #一个系统级的prompt，用于指定chatgpt的人格
  prompt = prompt.format(question)
  response = request_api(prompt)
  return response 

def list_detail(entity,content):  #将实体的信息一条条列出
  prompts = '{} Please list all features of {} which mentioned in above with number, do not include {} in your list.'
  prompts = prompts.format(content,entity,entity)
  response = request_api(prompts)
  return response 

def entity_conform_to_detail(detail):                           
  prompts = 'You should find an entity conform to the following describtion: {}. If you fail to find a perfect match, please say an entity that mathes the requirements as much as possible. You need to give the percentage of the entity that meets requirements.'
  prompts = prompts.format(detail)
  response = request_api(prompts)
  return response 

def request_api(Prompts):          #把所有的api访问集中在一个函数
  flag = True
  while flag:
      try:
          response = openai.ChatCompletion.create(
              model="gpt-3.5-turbo",
              messages=[{"role": "user", "content": Prompts}],
              max_tokens=256,
              temperature=0           
            )
          flag = False
      except Exception as e:
          print("try again!")
          print(e)
          time.sleep(5)
  text_response = response["choices"][0]["message"]["content"]
  cost = response["usage"]["total_tokens"]

  return text_response


def question_generation_pipeline(entity, content):      #RV-QG variants
  # content = main_chat(entity)       #生成关于entity的content
  question = question_generation(entity,content)    #construct query
  answer = reverse_modeling(question)   #access databases
  record = {'entity':entity,'claim':content,'question':question,'answer':answer}  
  
  return record

def entity_matching_pipeline(entity,content):      #RV-EM variants
    detail = list_detail(entity,content)      #list requirements
    answer = entity_conform_to_detail(detail) #return entity that match the requirements and report percentage
    record = {'entity':entity,'claim':content,'detail':detail,'answer':answer}
     
    return record
  

  
if __name__ == '__main__':
  record = question_generation_pipeline('Proclamation 10043')
  
