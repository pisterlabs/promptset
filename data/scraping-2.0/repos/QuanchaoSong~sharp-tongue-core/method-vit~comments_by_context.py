from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import openai


OPENAI_API_KEY = "Your OpenAI API Key"

class Comments_By_Context:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float32
        )
        self.model.to(self.device)
        
        openai.api_key = OPENAI_API_KEY

    def analyse_image_url(self, image_url):
        the_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        return self.__analyse_image(image=the_image)
    
    def analyse_image_data(self, image_data):
        the_image = Image.open(image_data).convert("RGB")
        return self.__analyse_image(image=the_image)
    
    def __analyse_image(self, image):
        context_sentence = self.__get_context_of_image(image)
        sacarstic_comment = self.__generate_sacarstic_comment_to_sentence(context_sentence)
        return [sacarstic_comment]

    def __get_context_of_image(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device, torch.float32)
        generated_ids = self.model.generate(**inputs, max_new_tokens=50)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text
    
    def __generate_sacarstic_comment_to_sentence(self, sentence):
        prompt = f"Generate a sarcastic comment to this sentence:\"{sentence}\"."
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # print("response:", response)
        choices = response["choices"]
        answer_item = choices[0]
        comment = answer_item["text"].strip()
        return comment
    

if __name__ == '__main__':
    tool_for_context = Comments_By_Context()
    res = tool_for_context.analyse_image_url("http://n.sinaimg.cn/sinacn15/250/w640h410/20180318/6d63-fyshfur2581706.jpg")
    print("res:", res)