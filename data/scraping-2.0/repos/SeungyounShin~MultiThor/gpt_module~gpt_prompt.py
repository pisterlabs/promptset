# Import necessary packages
import os,re
import openai
from retrying import retry

# Set the OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

@retry(stop_max_attempt_number=5, wait_exponential_multiplier=1000, wait_exponential_max=10000)
def query_gpt4(messages: list):
    """Function to query GPT-4 with a list of messages"""
    # Call the OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Use "gpt-4" as the model
        messages=messages
    )

    # Extract the response content and status code
    content = response["choices"][0]["message"]["content"]
    status_code = response["choices"][0]["finish_reason"]

    return content, status_code

class GPT4Query:
    def __init__(self, model="gpt-4", role_msg = 'You are a helpful assitant.'):
        self.model = model
        self.messages = [{"role": "system", "content": f'{role_msg}'}]  # initial system message
        self.response = None

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def query(self, user_input):
        self.add_message("user", user_input)
        self.response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages
        )
        self.add_message("assistant", self.get_response_content())  # save the assistant's response
        return self.response

    def get_response_content(self):
        if self.response:
            return self.response["choices"][0]["message"]["content"]
        else:
            return None

    def get_response_status(self):
        if self.response:
            return self.response["choices"][0]["finish_reason"]
        else:
            return None

class GPT4Agent(GPT4Query):
    def __init__(self, model="gpt-4", role_msg = 'You are a helpful assistant.'):
        super().__init__(model, role_msg)
        
        self.last_aciton = None

    def act(self, user_input):
        self.add_message("user", user_input)
        self.response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
            stop = ['###','\n']
        )
        response = self.get_response_content()
        act_type, content = self.parse_response(response)
        response_fmt = f'<{act_type}>{content}</{act_type}>'
        self.last_aciton = content
        self.add_message("assistant", response_fmt)  # save the assistant's response

        return act_type, content

    def parse_response(self, response):
        #print(response) # debug
        tags = re.findall(r'<(.*?)>', response)[:2]
        content = re.findall(r'<.*?>(.*?)</.*?>', response)

        #print(tags)
        # delete duplicating agent
        if tags[0]=='action':
            content_split = [i.strip() for i in content[0].split(',')]
            unique_commands = []
            seen_agents = set()
            for command in content_split:
                parts = [i.strip() for i in command.split(':')]
                if len(parts) == 2:
                    agent, _ = parts
                    if agent not in seen_agents:
                        unique_commands.append(command)
                        seen_agents.add(agent)
            
            content_fixed = ', '.join(unique_commands)
            return tags[0], content_fixed
    
        return tags[0], content[0]

    
    def getLastAction(self):
        return self.last_aciton

    
if __name__=='__main__':
    # Create a GPT4Query instance
    gpt4_query = GPT4Query()

    while True:
        # Get user input
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        # Query GPT-4
        gpt4_query.query(user_input)

        # Print the response content
        print(f"GPT-4: {gpt4_query.get_response_content()}")