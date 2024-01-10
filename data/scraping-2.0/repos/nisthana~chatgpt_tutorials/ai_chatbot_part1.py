import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
message_history = []
message_to_send = ""
while (message_to_send != 'goodbye'):
    message_to_send = input("Enter message to send:")
    user_message = {"role":"user","content": message_to_send}
    message_history.append(user_message)
    user_message_formatted = ">> You asked: {user_message}".format(user_message = message_to_send)
    print(user_message_formatted)
    chat_completion = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=message_history)
    choices = chat_completion['choices']
    chatgpt_response = choices[0]['message']
    content = chatgpt_response['content']
    chatbot_response = {"role":"assistant","content": content}
    message_history.append(chatbot_response)
    chatgpt_response_formatted = ">> ChatGPT answered : {chatgtp_response}".format(chatgtp_response = content)
    print(chatgpt_response_formatted)









