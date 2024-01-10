import openai
import nltk
from nltk import word_tokenize
from autocorrect import Speller
from .TreatSentence import NER
import json
from transformers import T5ForConditionalGeneration, T5Tokenizer
import random
from abc import *

class TextProcessing :

    def __init__(self, ner_model) :
        self.ner = NER(ner_model)
        self.material_example = 'Wood, Metal, Plastic, Glass, Fabric, Leather'
        self.color_exmaple = 'Turquoise, Teal, Navy blue, Sky blue, Royal blue, Cobalt blue'
        self.size_exmaple = 'Tiny, Small, Petite, Long, Extensive, Far-reaching'
        self.design_exmaple = 'Modern, Contemporary, Minimalist, Scandinavian, Mid-century modern, Industrial'
        

    @abstractmethod
    def request(self, prompt, num, model_name) :
        pass

    
    def checkPrompt(self, prompt) :
        if len(prompt.split()) == 1 :
            print('It\'s too short.')
            return 0
        
        objs = self.ner.get_missing_tags(prompt)
        
        if 'Furniture' in objs:
            print('do not have furniture')
            return -1
        
        recommend_dict = {'Material': self.material_example, 'Color': self.color_exmaple, 'Size': self.size_exmaple, 'Design': self.design_exmaple}
        express = ''.join('-' + '<span style=\"color:red\"><b>' + o + '</b></span>: ' + recommend_dict[o] + '\n\n' for o in objs)
        recommend = 'How about adding these properties?\n\n'+  express
        print('recommend')
        print(recommend)
        
        return recommend
    
    @abstractmethod
    def getAnswer(self, prompt) :
        pass
    
class TextProcessing_gpt(TextProcessing) :

    def __init__(self, ner_model, key) :
    #사용시 키 입력
        openai.api_key = key
        super().__init__(ner_model)
        

    def request(self, prompt) :
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You should help generate furniture text for stable diffusion."},
            {"role": "user", "content": prompt},
        ]
    )

        return response.choices[0].message.content
        
       

    #사용해야 하는 것 / input : 사용자 input, output : json
    def getAnswer(self, prompt) :
        
        recommend = super().checkPrompt(prompt)
        
        if recommend == -1 or recommend == 0:
            return recommend
        
        json_object = {
                    "recommend" :recommend
                }
        
        input_text = "The sentence that supplements and explains \"A white chair\" is as follows: \n\
    - White Office Chair with Adjustable Handles: This sleek white office chair features padded armrests that can be adjusted to fit the user\'s preference. The seat and backrest are also padded for comfort and support during long work hours. The chair\'s height can be adjusted with a pneumatic lever, and it sits on smooth-rolling casters for easy mobility.\n\
    - White Rocking Chair with Wide Armrests: This charming white rocking chair has wide, curved armrests that are perfect for resting a beverage, book, or tablet. The chair is made of solid wood and has a contoured seat and backrest for maximum comfort. The gentle rocking motion is perfect for relaxing, reading, or napping.\n\
    - White Dining Chair with Upholstered Handles: This elegant white dining chair features upholstered armrests for added comfort during long meals or social gatherings. The chair\'s sleek, modern design complements any decor, and the sturdy metal frame ensures durability. The seat and backrest are also upholstered in easy-to-clean fabric, making this chair a practical choice for daily use.\n\
    So, enhance the following sentence, and provide a detailed explanation about this sentence: \"" + prompt + "\" \nGive 3 examples. The format is as follows.\n\
        -sentence : detail\n\
        -sentence : detail\n\
        -sentence : detail\n"
        
        verifiedSentence = prompt
        while (True) :
            try :
                response = self.request(input_text)
                print('response: ')
                print(response)
                detail_list = [re for re in response.split('\n') if re[0] == '-' or re[0].isdigit()]

                if len(detail_list) < 3 :
                    raise Exception("format error")
                index = 1
                detail_json = {}
                detail_json['detail0'] = {"prompt" : verifiedSentence, "detail" : "Original Prompt"}
                for d in detail_list :
                    detail_json['detail' + str(index)] = {"prompt" : d.split(': ')[0], "detail" : d.split(': ')[1] + '\n'}
                    index += 1
                break
                
            except Exception as e:
                print('error: ')
                print(e)
                continue

  
        json_object['detail'] = detail_json
        json_string = json.dumps(json_object)
        return json_string, response
    
    
    
    
class TextProcessing_T5(TextProcessing) :

    def __init__(self, ner_model, model_name) :
    #사용시 키 입력
#         self.model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
        super().__init__(ner_model)
        

    def request(self, prompt) :
        result = []

        for _ in range(3) :
            # prepare for the model
#             input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to("cuda")
            input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids
            # generate
            outputs = self.model.generate(input_ids, min_length=random.randrange(50, 60), max_length = random.randrange(80, 100), num_beams=random.randrange(1, 50), repetition_penalty=random.randrange(1, 4) / 10, temperature = random.randrange(8, 11) / 10, no_repeat_ngram_size = 1)
            result.append(self.tokenizer.decode(outputs[0], skip_special_tokens=True))

        return result

    def getAnswer(self, prompt) :
        
        recommend = super().checkPrompt(prompt)
        
        if recommend == -1 or recommend == 0:
            return recommend
        
        json_object = {
                    "recommend" :recommend
                }
        
        input_text = "Your role is a designer who explains furniture designs to users.\n\
                            The sentence that supplements and explains \"A white chair\" is as follows:\n\n\
                            - A white leather lounge chair with large wooden arms: This chair has a sleek and modern white leather upholstery that complements its wooden arms. The arms have a large size, ensuring comfortable and secure seating. The chair has a plush padded seat and backrest, ensuring optimal comfort and support.\n\n\
                            Enhance the following sentence, and provide a detailed explanation about this sentence\n\n\"" + prompt + "\"\n\n\
                            The format is as follows.\n\n\
                            - sentence : detail"
        
        
        verifiedSentence = prompt
        while (True) :
            try :
                response = self.request(input_text)
                print('response: ')
#                     if response[0].split(': ')[0] == response[1].split(': ')[0] or response[0].split(': ')[0] == response[2].split(': ')[0] or response[1].split(': ')[0] == response[2].split(': ')[0] :
#                         raise Exception('duplication')
                for r in response :
                    if len(r.split(": ")) > 3:
                        raise Exception("format error")
                    print(r)
                    print('-'*50)
                detail_list = [re for re in response if re[0] == '-' or re[0].isdigit()]
                
                index = 1
                detail_json = {}
                detail_json['detail0'] = {"prompt" : verifiedSentence, "detail" : "Original Prompt"}
                for d in detail_list :
                    detail_json['detail' + str(index)] = {"prompt" : d.split(': ')[0], "detail" : d.split(': ')[1] + '\n'}
                    index += 1
                break
            except Exception as e:
                print(e)
                continue

  
        json_object['detail'] = detail_json
        json_string = json.dumps(json_object)
        return json_string, response