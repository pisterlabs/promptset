import json
import openai
import rospy

class FastChat:
    def __init__(self):
        openai.api_key = 'EMPTY'
        openai.api_base = rospy.get_param("/offline_conversation/fastchat/api_base", 'http://localhost:6000/v1')
        self.prompts = []
        self.model = rospy.get_param("/offline_conversation/fastchat/model", 'vicuna-7b-v1.3')
        self.memory_size = rospy.get_param("/offline_conversation/fastchat/memory_size", 5)
        self.max_token_length = rospy.get_param("/offline_conversation/fastchat/max_token_length", 4096)
        self.character = rospy.get_param("/offline_conversation/fastchat/character", "qtrobot")        
        self.system_prompt = rospy.get_param("/offline_conversation/fastchat/prompt", "")

    def create_prompt(self, user_prompt):
         # cut off long input
        if len(user_prompt) > self.max_token_length:
            user_prompt = user_prompt[:self.max_token_length]

        if not self.prompts:
            self.prompts.append({"role": "system", "content": self.system_prompt})
        self.prompts.append({'role': 'user', 'content': user_prompt})

        if self.memory_size > 0 and len(self.prompts) > self.memory_size:
            self.prompts.pop(1)
        # ensure the prompt size remain under max_token_length
        while len(json.dumps(self.prompts)) > self.max_token_length:
            self.prompts.pop(1)
        return self.prompts


    def generate(self, message, sentence_callback=None):
        try:
            completion = openai.ChatCompletion.create(
                model=self.model,
                messages=self.create_prompt(message),
                stream=True,
                temperature=rospy.get_param("/offline_conversation/fastchat/temperature", 0.8),
                # max_tokens=rospy.get_param("/offline_conversation/fastchat/max_tokens", 41),
                frequency_penalty=rospy.get_param("/offline_conversation/fastchat/frequency_penalty", 0.6),
                presence_penalty=rospy.get_param("/offline_conversation/fastchat/presence_penalty", 0.6)
                )
            
            max_sentence = rospy.get_param("/offline_conversation/fastchat/max_sentence", 0)
            response = ""
            text_all = ""
            sentence_count = 0
            stopped = False
            for chunk in completion:
                delta = chunk["choices"][0]["delta"]
                content = delta.get('content')
                if content:
                    response = response + content    
                    text_all = text_all + content
                sentence, response = self.extract_sentence(response)
                if sentence and sentence_callback:
                    sentence_callback(sentence)
                    sentence_count = sentence_count + 1
                    if max_sentence > 0 and sentence_count >= max_sentence:
                        stopped = True
                        break                    
            if not stopped and response and sentence_callback:
                sentence_callback(response)
            self.prompts.append({"role": "assistant", "content": text_all})
            return text_all
        except Exception as e:
            print(e)
            return None

    def extract_sentence(self, buff):
        deliemeters = ".!?;\n"
        index = None
        for char in buff:
            if char in deliemeters:
                index = buff.index(char)
                break
        if index == None:
            return None, buff

        extracted = buff[:index+1]
        extracted = extracted.strip().replace('*', '').replace('/', ' ')
        remaining = buff[index+1:]
        remaining = remaining.strip().replace('*', '').replace('/', ' ')
        return extracted, remaining
