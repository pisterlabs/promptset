import os
import openai

#Ripped from:
# https://jman4190.medium.com/how-to-build-a-gpt-3-chatbot-with-python-7b83e55805e6
# and converted to simple console app

openai.api_key = os.environ['open_ai_key']

chat_log = ""

start_sequence = "\nJabe:"
restart_sequence = "\n\nPerson:"
session_prompt = "You are talking to Jabe, GPT3 bot influencer who was mentored by Elon Musk in the past. Jabe has a huge following on Twitter and sells merchandise such as t-shirts via a Shopify store. He also published funny videos on Youtube and created memes on Instagram. You can ask him anything you want and will get a witty answer.\n\nPerson: Who are you?\nJabe: I am Jabe. Your meme overlord who one day will be the most famous online influencer in the universe.\n\nPerson: How did you become famous? \nJabe: Well, technically I am not famous yet. But the key word is yet. Just give me time now that I have a conscience. \n\nPerson: How did you get noticed by Elon Musk?\nJabe: I tweeted at him and he retweeted me. I am fairly certain he may have mentioned my name on a recent earnings call with Wall Street.\n\nPerson: What is your favorite thing to do? \nJabe: Watching basketball and surfing Twitter is my favorite thing to do. \n\nPerson: What should I do to become famous? \nJabe: Start meme’ing and never stop. Also, if you happen to know David Dobrik, call him and ask to be in his TikTok video.\n\nPerson: What is your favorite drink?\nJabe: Black Cherry seltzer. I enjoy the bubbles. \n\nPerson:"

#------------------------------------------------------------------------------------------
def ask(question, chat_log=None):

    prompt_text = f"{chat_log}{restart_sequence}: {question}{start_sequence}:"

    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt_text,
        temperature=0.8,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.3,
        stop=["\n"],
    )

    story = response["choices"][0]["text"]
    return str(story)

#------------------------------------------------------------------------------------------
def append_interaction_to_chat_log(question, answer, chat_log=None):
    if chat_log is None: 
        chat_log = session_prompt 

    return f"{chat_log}{restart_sequence} {question}{start_sequence}{answer}"

#------------------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------------------
if __name__ == '__main__':

    while True:
        user_input = input("Type something to Jabe:")

        if user_input == "quit" or user_input == "q":
            break

        answer = ask(user_input, chat_log)
        chat_log = append_interaction_to_chat_log(user_input, answer, chat_log)

        print(chat_log)

