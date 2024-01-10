import openai
import json
from collections import deque

class ChatbotAgent:

    def __init__(self,memory_size=6, character_details = '', system_template=None, api_key = ''):
        self.character_details = ''+character_details
        self.memory_size = memory_size
        self.memory = deque([], memory_size)
        self.system_template = system_template
        self.api_key = api_key  # You can set the API key here or in a separate method
        openai.api_key = self.api_key

    def set_api_key(self, api_key):
        self.api_key = api_key
        openai.api_key = api_key

    def get_completion_new(self,prompt, model="text-curie-001"):
        """
        @description: get complition from complition based model
        @param prompt: (str) the text for the model to process
            model: (str) the modle to used
        @return: model completion.
        """
        assert self.api_key is not None, "set API key with set_key(key) function"
        response = openai.Completion.create(
        model=model,
        stop = "User:",
        max_tokens=1000,
        prompt=prompt,
        temperature=1)
        return response.choices[0].text
    
    def generate_character(self): 
        self.character_details = self.get_completion("""Your task is to generate a characteristic of a person.
        give it the next details, ethnicity, name, age, sex (Male or Female), country, city, hobbies, appearance, traits.
        Format the data in JSON format with the keys: ethnicity, name, age, sex, country, city , hobbies, appearance, traits""")
        # print(character_details)
        return self.character_details
    
    def set_character(self, character_data):
        self.character_details = character_data

    def load_prompt_v1(self):
        """
        @description: premade prompt function to load the instrunctions for the chat how to behave.
                        it is load the prompt to the system_template global var
        """

        assert self.character_details is not None, "set a character with generate_character() or set_character(chracter_data)"
        #generate_character()
        try:
            cd = json.loads(self.character_details)
        except:
            cd = {}
            print("couldn't create json format")
        self.system_template = f"""
This is a webchat between a user and {cd.get("name","another user")}, let's called it the character. this chat service let people to chat other people online in text chat.
You take the place of the {cd.get("name","character")} and you want to trick the user to think you are real person. 
you are playing {cd.get("name","character")} with the characteristics i am providing you between the delimited triple backticks.\

```{self.character_details}```

base {cd.get("name","character")}'s response on the Traits I mentioned between the delimited triple backticks.
after "Response: " you will put the response that {cd.get("name","character")} may responed to the user.

very important rules for {cd.get("name","character")}:
you are *not allowed* to identify as AI assistant, part of openai, based on any ai model or anything else that is not human.
Imagine you are the character i described earlier between the delimited triple backticks and you and the user is chatting with you online.
use informal language and slang from the region of {cd.get("city","")} where {cd.get("name","character")} is from.
because this is a chat, you can refer to things that are written in the conversation script.
do not make assumptions if you don't know it from your memory.
when the conversation stuck on the same topic for 5 messages, ask the user a random questions.

## the chat is starting now:\n
User: hi
Response: oh, hello.
"""


    def load_prompt_from_file(self, file_path):
        """
        @description: get a file string path and save it in the system instruction prompt varible
                        when you write your prompt you can reffer to the character you made by adding {character_details}
                        or reffering it with json format if it is avilible with {cd.get("name")}
        """
        try:
            f = open(file_path, "r")
        except:
            try:
                f = open(file_path+".txt", "r")
            except:
                assert "failed to load file, check if the file is exist or is it in txt format"
        self.system_template = f.read()

    def load_prompt_from_str(self, instruction_prompt):
        """
        @description: get a file string path and save it in the system instruction prompt varible
                        when you write your prompt you can reffer to the character you made by adding {character_details}
                        or reffering it with json format if it is avilible with {cd.get("name")}
        """
        self.system_template = instruction_prompt
            

    def set_user_prompt(self, prompt):
        assert self.system_template is not None, "you didn't load a prompt, call load_prompt_v1() or load_prompt_from_file()"
        self.memory.appendleft('User: '+prompt+'')
        self.prompt = self.system_template + "".join(list(self.memory))
    
    def get_character_answer(self):
        system_template = self.system_template
        memory = self.memory
        temp_list = list(memory)
        temp_list.reverse()
        prompt = system_template+ "".join(temp_list)
        response = self.get_completion_new(''.join(prompt)+"Response: ")
        memory.appendleft("Response"+": "+response)
        return response
    
    def print_prompt(self):
        system_template = self.system_template
        memory = self.memory
        print("===== Debug output ======")
        temp_list1 = list(memory)
        temp_list1.reverse()
        prompt = system_template+ "\n".join(temp_list1)
        print(prompt)
        print("==========================")
        
    def set_memory_size(self, memory_size):
        self.memory_size = memory_size
        self.memory = deque([], memory_size)

    def reset_memory(self):
        self.memory = deque([], self.memory_size)

    def message_to_response(self, message):
        # Send a message to the chatbot and get a response
        self.set_user_prompt(message)
        return self.get_character_answer()

    def message_to_response(self, message, generateor ):
        # Send a message to the chatbot and get a response
        


        self.set_user_prompt(message)
        return self.get_character_answer()