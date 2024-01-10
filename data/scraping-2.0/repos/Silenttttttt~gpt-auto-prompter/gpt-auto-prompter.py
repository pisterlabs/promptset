import os
import openai
import json
import time


api_key = "sk-"



class Chatbot:
    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model

    def chat_completion_api(self, conversation):
        openai.api_key = self.api_key

        messages = [{"role": message["role"], "content": message["content"]} for message in conversation.messages]

        while True:
            try:
                response = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                messages=messages
                                ##rest of model arguments
                )
                content = response['choices'][0]['message']['content']

                conversation.add_message("assistant", content)
                return {"response": content}
            except openai.error.RateLimitError:
                print("Rate limit error encountered. Waiting for 30 seconds before retrying...")
                time.sleep(30)


#self.handle_error_response
class Conversation:
    def __init__(self):
        self.messages = []

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def read_from_json(self, filename):
        try:
            with open(filename, "r") as f:
                conversation_json = json.load(f)
            self.messages = conversation_json["messages"]
        except:
            pass
       # print(self.messages)

    def write_to_json(self, filename):
        conversation_json = {"messages": self.messages}
        with open(filename, "w") as f:
            json.dump(conversation_json, f, indent=2)

    def get_conversation_format(self):
        return [{"role": message["role"], "content": message["content"]} for message in self.messages]




def get_multiline_input(prompt, end_word):
    lines = []
    print(prompt)
    while True:
        line = input()
        if line.strip() == end_word:
            break
        lines.append(line)
    print("Sent message to API...")
    return '\n'.join(lines)

def autoprompt(conversation, chatbot, filename, feedback_chatbot, auto_prompt_sys_message, auto_prompt_message_limit):
    last_n_messages = conversation.messages[-auto_prompt_message_limit:]
    
    feedback_conversation = Conversation()
    feedback_conversation.add_message("system", auto_prompt_sys_message)
    
    for message in last_n_messages:
        if message["role"] != "system":
            feedback_conversation.add_message(message["role"], message["content"])
    num_tokens = sum(len(msg["content"]) for msg in feedback_conversation.messages) // 4
    print(f"Number of tokens before response: {num_tokens}")
    feedback_response = feedback_chatbot.chat_completion_api(feedback_conversation)
    feedback_content = feedback_response["response"]
    print("Auto prompter:")
    print(feedback_content)
    conversation.add_message("user", feedback_content)


    
    num_tokens = sum(len(msg["content"]) for msg in conversation.messages) // 4
    print(f"Number of tokens before response: {num_tokens}")
    
    response = chatbot.chat_completion_api(conversation)
    content = response["response"]
    conversation.write_to_json(filename)
    
    return content


def interact_chat(conversation, chatbot, filename, sys_message=None, auto_prompt=False, auto_prompt_sys_message=None, auto_prompt_message_limit=2, feedback_chatbot=None):
    try:
        if sys_message == '':
            sys_message = "You are a very helpful bot, you give very technical and step by step detailed solutions for the problems"
        if auto_prompt_sys_message == '':
            auto_prompt_sys_message = "You are the autoprompter, you strictly only talk as if you were the user, you need to help the user get his message across to the chatbot and make his intentions as clear as possible. You add to the question, not to the answer: "

        conversation.add_message("system", sys_message)
        while True:
            print("---")

            conversation.read_from_json(filename)

            user_input = get_multiline_input("Enter your message: ", "|")
            conversation.add_message("user", user_input)

            num_tokens = sum(len(msg["content"]) for msg in conversation.messages) // 4
            print(f"Number of tokens before response: {num_tokens}")

            response = chatbot.chat_completion_api(conversation)
            content = response["response"]

            print(f"Bot: {content}")
            print("---")

            if auto_prompt:
                while True:
                    content = autoprompt(conversation, chatbot, filename, feedback_chatbot, auto_prompt_sys_message, auto_prompt_message_limit)
                    print(f"Bot: {content}")
                    print("---")
                    time.sleep(1)

            num_tokens = sum(len(msg["content"]) for msg in conversation.messages) // 4
            print(f"Number of tokens after response: {num_tokens}")

            # Ask user if they want to update the conversation file
            #  update_file = input("don't update file? (y/n): ")
       #    if update_file.lower() != "y":
            conversation.write_to_json(filename)

            time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted by user.")

def main(api_key):

    conversation_name = input("Enter conversation name: ")
    conversation_filename = f"{conversation_name}.json"

    if input("Do you want to create a new conversation or load an existing one? (c/l): ") == "c":
        conversation = Conversation()
    else:
        conversation = Conversation()
        conversation.read_from_json(conversation_filename)

   
    model = "gpt-3.5-turbo"

    chatbot = Chatbot(api_key, model)
    feedback_chatbot = Chatbot(api_key, model)
    sys_message = None
    sys_message = input("What is the system message? : ")

    auto_prompt = input("Do you want to enable auto-prompting? (y/n): ").lower() == "y"
    auto_prompt_sys_message = None
    
        

    if auto_prompt:
        auto_prompt_sys_message = input("What is the system auto-prompt message?: ")
        try:
            auto_prompt_message_limit = int(input("How many messages should the auto-prompt use?: "))
        except:
            auto_prompt_message_limit = 3
            print(f"Got error, using default: {auto_prompt_message_limit}")
            

    interact_chat(conversation, chatbot, conversation_filename, sys_message, auto_prompt, auto_prompt_sys_message, auto_prompt_message_limit, feedback_chatbot)

   

    conversation.write_to_json(conversation_filename)



if __name__ == "__main__":
    main(api_key)
