import openai
import numpy as np
import imageio

class GPT_model():
  def __init__(self):
    openai.api_key = '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'
    self.model = 'gpt-3.5-turbo'
    self.input_type = ['text']
    self.output_type = ['text']
    self.description = 'language_model'
    self.model_label = 'gpt'

  def predict(self, inputs, system_message='', history=[]):
    message = inputs[0]
    response = openai.ChatCompletion.create(
      model=self.model,
      messages=self.get_messages(history, message, system_message)
    )
    answer = response['choices'][0]['message']['content']

    return [answer]

  def get_messages(self, history, message, system_message):
    messages = []
    for hist in history:
      if hist['answerer'] == self.model_label:
        text = ''
        for data, data_type in hist['input']:
          if data_type == 'text':
            messages.append({'role': "user", 'content': data})
            break 
          
        for data, data_type in hist['output']:
          if data_type == 'text':
            messages.append({'role': "user", 'content': data})
            break
      
      if 'input_disc' in hist:
        messages.append({'role': "user", 'content': hist['input_disc']})
      if 'output_disc' in hist:
        messages.append({'role': "assistant", 'content': hist['output_disc']})
        
    messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": message})
    return messages

  
if __name__ == 'main':
  model = GPT_model()
  print(model.render('hello'))