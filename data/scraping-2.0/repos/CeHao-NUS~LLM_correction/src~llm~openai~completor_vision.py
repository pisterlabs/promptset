from src.llm.openai.completor_openai import OpenAICompletor
import base64
import requests
from openai import OpenAI

class OpenAICompletorVision(OpenAICompletor):

    def __init__(self, api_key):
        self.messages = []
        self.client = OpenAI(api_key=api_key)
        self.headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
        }

    def answer_with_image(self, question, image_paths):
        self._add_image_question(question, image_paths)
        ans = self._get_completion()
        self._add_answer(ans)
        return self._last_message()
    
    def _add_image_question(self, question, image_paths):
        
        base64_images = [encode_image(image_path) for image_path in image_paths]
        content = [{"type": "text", "text": question} ] \
            + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"} } for base64_image in base64_images]
            
        self.messages.append({'role':'user', 'content':content})

    def _get_completion(self):
        response = self.client.chat.completions.create(
        model = 'gpt-4-vision-preview',
        messages = self.messages,
        temperature = 0,
        max_tokens=4096,
        )
        # print(response.usage)
        return response.choices[0].message.content
        # print(type(response))

        # print(response.choices[0].message.content)


    # def _get_completion(self):
    #     payload = {
    #     "model": "gpt-4-vision-preview",
    #     "messages": self.messages,
    #     "max_tokens": 4000,
    #     "temperature": 0
    #     }
    #     response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)
    #     print(response.json())
    #     print(response.json()['usage'])
    #     return response.json()['choices'][0]['message']['content']

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
if __name__ == "__main__":
    api_key = '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'
    completor = OpenAICompletorVision(api_key)
    question = "What's in this image?"
    image_path = "./zeroshot_test/Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    ans = completor.answer_with_image(question, image_path)
    print(ans)