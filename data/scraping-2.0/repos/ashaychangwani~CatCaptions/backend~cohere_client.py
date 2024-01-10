import os
import cohere

class CohereClient:
  def __init__(self):
    self.co = cohere.Client(os.getenv('COHERE_API_KEY'))
    self.base_prompt = 'Generate 5 funny insta captions from the point of view of the pet for a picture of: \"{}\"\n\nSeparate the captions with a newline, number them and use \" before and after the captions'
    
  def create_captions(self, desc):
    response = self.co.generate(
      model='command-xlarge-beta',
      prompt=self.base_prompt.format(desc),
      max_tokens=176,
      temperature=3,
      k=0,
      p=0.75,
      frequency_penalty=0,
      presence_penalty=0,
      stop_sequences=[],
      return_likelihoods='NONE')
    captions = response.generations[0].text
    print('Prediction: {}'.format(captions))
    return captions
  
if __name__ == '__main__':
    co = CohereClient()
    co.create_captions('a dog running with a toy in its mouth')
  