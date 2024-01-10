import os
import openai

class GPTClient:
  def __init__(self):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    self.base_prompt = 'Generate 5 funny insta captions from the point of view of the pet for a picture of: \"{}\"\n\nSeparate the captions with a newline, number them and use \" before and after the captions'
    
  def create_captions(self, desc):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=self.base_prompt.format(desc),
        temperature=0.82,
        max_tokens=400,
        top_p=1,
        frequency_penalty=1,
        presence_penalty=1
    )
    print(response)
    captions = response["choices"][0]["text"]
    print('Prediction: {}'.format(captions))
    return captions
  
if __name__ == '__main__':
    co = GPTClient()
    co.create_captions('a dog running with a toy in its mouth')
  