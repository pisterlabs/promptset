import openai
import savedfile
from function_list import function_list
from regex import regex as re
from sql_queries import Sql_Query
from mediagetter import downloader
from start import Video_Editor
import json
import dictfun


class Quote_Getter(Video_Editor,downloader):
  def __init__(self,query):
    super().__init__()
    self.query = query
  
  def query_ai(self):
    openai.api_key = savedfile.openai_key()
    completion = openai.ChatCompletion.create( 
      model="gpt-3.5-turbo-0613",
      messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{self.query}"},],
      functions=function_list())

    response = completion.choices[0].message
    return response
  
  def quote_ai(self,qu):
    openai.api_key = savedfile.openai_key()
    completion = openai.ChatCompletion.create( 
      model="gpt-3.5-turbo-0613",
      messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{qu}"},],
      )

    response = completion.choices[0].message
    return response
  
  def query_sort(self):
    response=self.query_ai()
    print(response)
    try:
      regex_arguments = re.sub(r"\t|\n|\r", '', response["function_call"]["arguments"])
      regex_arguments = json.loads(regex_arguments)
      
      function_call = response["function_call"]
      function_name = function_call["name"]
      print(regex_arguments)
      data = eval(dictfun.dict_fun(function_name))
      return data

    except:
      return response["content"]
    
  def answer_back(self):
    data = self.query_sort()
    return data
  
  def add_quotes(self,amount):
    data = self.quote_ai(f"Get me {amount} motivational quotes")
    if  int(amount) > 1:
      data_list = list(map(lambda x: x[1],data["content"].split("\n")))
      print(data_list)
      return self.insert_quotes_auto(data_list)
    print([data['content']])
    return self.insert_quotes_auto([f"{data['content']}"])
  
  

