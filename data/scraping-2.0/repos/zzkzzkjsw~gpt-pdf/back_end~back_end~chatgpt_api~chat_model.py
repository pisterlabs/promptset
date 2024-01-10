import openai
from .utils.chatgpt_utils import num_tokens_from_messages

class ChatModel:
    '''
    add_system_message
    add_example_question
    add_example_answer

    trim_conversation
    '''

    def __init__(self):

        # 设置 API Key，申请地址：https://platform.openai.com/account/api-keys
        openai.api_key = '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' 
        # 设置组织，查看地址：https://platform.openai.com/account/org-settings
        openai.organization = "org-jy2eL6AhJYOSnDydGNQpPcro"
        self.token_limit = 4096
        self.max_response_tokens = 250

        self.conversation = []
        self.num_fixed_conversation = 0
        self.num_tokens = 0
        self.num_fixed_tokens = 0

    def add_system_message(self, system_message="You are a helpful assistant."):
        system_message = {"role": "system", "content": system_message}
        self.conversation.append(system_message)
        self.num_fixed_conversation += 1
        num_message_tokens = num_tokens_from_messages([system_message])
        self.num_tokens += num_message_tokens
        self.num_fixed_tokens += num_message_tokens

    def add_example_question(self, example_question):
        example_message = {
            "role": "system",
            "name": "example_user",
            "content": example_question
            # "content": "New synergies will help drive top-line growth."
        }
        self.conversation.append(example_message)
        self.num_fixed_conversation += 1
        num_message_tokens = num_tokens_from_messages([example_message])
        self.num_tokens += num_message_tokens
        self.num_fixed_tokens += num_message_tokens
    
    def add_example_answer(self,example_answer):
        example_message = {
            "role": "system",
            "name": "example_assistant",
            "content": example_answer
            # "content": "Things working well together will increase revenue."
        }
        self.conversation.append(example_message)
        self.num_fixed_conversation += 1

        num_message_tokens = num_tokens_from_messages([example_message])
        self.num_tokens += num_message_tokens
        self.num_fixed_tokens += num_message_tokens

    def add_user_question(self,question):
        message = {
            "role": "user",
            "content": question
        }
        self.conversation.append(message)
        num_message_tokens = num_tokens_from_messages([message])
        self.num_tokens += num_message_tokens


    def add_assistant_answer(self,answer):
        message = {
            "role": "assistant",
            "content": answer
        }
        self.conversation.append(message)
        num_message_tokens = num_tokens_from_messages([message])
        self.num_tokens += num_message_tokens
        

    def trim_conversation(self):
        num_convseration = len(self.conversation)
        num_flexible_conversation = num_convseration - self.num_fixed_conversation
        half = int(num_flexible_conversation/2)
        self.conversation = self.conversation[:self.num_fixed_conversation].extend(self.conversation[self.num_fixed_conversation+half:])

    def clear_conversation(self):
        self.conversation = self.conversation[:self.num_fixed_conversation]






        
            
            
        
                








