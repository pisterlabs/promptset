import openai
import os
from dotenv import load_dotenv
# import Listen, Speak  
openai.api_key = ""
load_dotenv()
completion = openai.Completion()


def Reply(question, chat_log=None):
    FileLog = open(r"Database\database.csv", "r")
    chat_log_template = FileLog.read()
    FileLog.close()

    if chat_log is None:
        chat_log = chat_log_template
    prompt = f'{chat_log}Q:  {question}\A:  '
    response = completion.create(   model="text-davinci-002",
                                    prompt=prompt,
                                    temperature=0.7,
                                    max_tokens=1024,
                                    top_p=0.5,
                                    frequency_penalty=0.5,
                                    presence_penalty=0
                                 )
    answer = response.choices[0].text.strip()
    chat_log_template_update = chat_log_template + \
        f'\nQ:  {question}, \nA:  {answer}'
    FileLog = open(r"Database/database.csv", "w")
    FileLog.write(chat_log_template_update)
    FileLog.close()
    return answer


if __name__ == "__main__":

    while True:

        que = str(input(" "))
        reply=Reply(que)
        print(reply)
        # reply=reply.replace(reply,"")
        
# from ChatterBot.chatterbot.chatterbot import ChatBot
# from chatterbot.trainers import ListTrainer

# # Create a new chatbot instance
# chatbot = ChatBot('Radiot')

# # Create a trainer to teach the chatbot with previous conversations
# trainer = ListTrainer(chatbot)

# # Train the chatbot with a few example conversations
# trainer.train([
#     'Hello',
#     'Hi there!',
#     'How are you doing?',
#     'I\'m doing great.',
#     'That is good to hear',
#     'Thank you.',
#     'You\'re welcome.'
# ])

# # Start the conversation with the chatbot
# while True:
#     user_input = input('You: ')
#     if user_input.lower() == 'bye':
#         print('Jarvis: Goodbye!')
#         break
#     response = chatbot.get_response(user_input)
#     print('Ra:', response)
