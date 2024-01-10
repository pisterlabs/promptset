import openai

class ChatGPT:
    def __init__(self, api_key, filepath):
        self.api_key = api_key
        self.short_term_memory = []
        self.system_prompt = self.open_file(filepath)
        openai.api_key = self.api_key

    def open_file(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as infile:
            return infile.read()

    def save_file(self, filepath, content):
        with open(filepath, 'a', encoding='utf-8') as outfile:
            outfile.write(content)

    def chat(self, user_input, temperature=0, frequency_penalty=0.2, presence_penalty=0, max_turns=10):
        self.short_term_memory.append({"role": "user","content": user_input})

        # Only use the last max_turns turns of the short_term_memory
        if len(self.short_term_memory) > max_turns * 2:
            self.short_term_memory = self.short_term_memory[-max_turns * 2:]

        messages_input = self.short_term_memory.copy()
        prompt = [{"role": "system", "content": self.system_prompt}]
        messages_input.insert(0, prompt[0])

        functions=[
                
                {
                    "name": "talk_to",
                    "description": "You will start a conversation with nearby players",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "listener_id": {
                                "type": "string",
                                "description": "Name of the player you would like to chat with",
                            },
                            "message": {
                                "type": "string",
                                "description": "Message you would like to send to the player",
                            }
                        },
                        "required": ["listener_name", "message"],
                    },
                },
                {
                    "name": "walk_to",
                    "description": "You wil walk all the way to the destination",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location_name": {
                                "type": "string",
                                "description": "Name of the place you would like to walk to",
                            }
                        },
                        "required": ["location_name"],
                    },
                },
                {
                    "name": "save_file",
                    "description": "Save the provided content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filepath": {
                                "type": "string",
                                "description": "Path to the file where the content will be saved",
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to be saved in the file",
                            },
                        },
                        "required": ["filepath", "content"],
                    },
                },
            ]

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            messages=messages_input,
            functions=functions,
            function_call="auto"
        )

        chat_response = completion['choices'][0]['message']

        print("CHAT RESPONSE1 :", chat_response)

        function_info = {}  
        response = ""  # Initialize response to an empty string

        if 'function_call' in completion.choices[0].message:
            function_info = completion.choices[0].message['function_call']
            response = function_info
            print("FUNCTION INFO :", function_info)
        
        self.short_term_memory.append({"role": "assistant", "content": response})

        print("RESPONSE :", response)

        return response

api_key = "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"
filepath = 'prompts/chatbot10.txt'

#Initialize class
chat_agent = ChatGPT(api_key, filepath)

#Initial prompt
user_message = "Hello Mr.PyDev. I am Ms.Python. Your goal is to walk to a location"
#Calling Agent
response = chat_agent.chat(user_message)
print("FINAL RESPONSE :", response)
