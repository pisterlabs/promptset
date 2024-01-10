from PIL import Image
import requests
import replicate
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


OPENAI_API_KEY = "Your OpenAI API Key"

REPLICATE_API_TOKEN = "Your Replicate API token"

class Comments_By_Context_Replicate:
    def __init__(self) -> None:
        self.k = 3

        self.MINI_GPT_MODEL = "joehoover/instructblip-vicuna13b:c4c54e3c8c97cd50c2d2fec9be3b6065563ccf7d43787fb99f84151b867178fe"
        self.replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)
        openai.api_key = OPENAI_API_KEY

    def analyse_image_url(self, image_url):
        return self.__analyse_image(image=image_url)
    
    def analyse_image_data(self, image_data):
        return self.__analyse_image(image=image_data)
    
    def __analyse_image(self, image):
        print("\n===============Parsing Context===============\n")
        context_sentence = self.__get_context_of_image(image)
        print("======Context sentence======")
        print(context_sentence)
        sacarstic_comment_list = self.__generate_sacarstic_comment_to_sentence(context_sentence)
        print("\nsacarstic_comment_list:", sacarstic_comment_list)
        return sacarstic_comment_list
    

    @retry(wait=wait_random_exponential(min=3, max=60), stop=stop_after_attempt(6))
    def __generate_sacarstic_comment_to_sentence(self, sentence):
        prompt = f"From different perspectives, generate {self.k} sarcastic comments towards the content of a picture: \"{sentence}\". It's better to give them a sarcastic tone. Give result in pure Python list like [\"comment1\", \"comment2\", \"comment3\"], adding escape mark if necessary."
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=3500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # print("response:", response)
        choices = response["choices"]
        answer_item = choices[0]
        comment_list_str = answer_item["text"].strip()
        if (comment_list_str.lower().startswith("result:")):
            comment_list_str = comment_list_str[len("result:"):].strip()
        comment_list = eval(comment_list_str)
        return comment_list   
    

    def __get_context_of_image(self, image):
        input_params = {
            "prompt": "Describe this picture in detail.",
            "img": image
        }

        return self.__run_replicate(self.MINI_GPT_MODEL, input_params)
    
    def __run_replicate(self, model, input_params):
        output = self.replicate_client.run(
            model,
            input=input_params
        )

        result = ""
        for item in output:
            # print(item)
            result += item
        # print("result:", result)
        return result
    

if __name__ == '__main__':
    tool_for_context = Comments_By_Context_Replicate()
    res = tool_for_context.analyse_image_url("http://n.sinaimg.cn/sinacn15/250/w640h410/20180318/6d63-fyshfur2581706.jpg")
    print("res:", res)