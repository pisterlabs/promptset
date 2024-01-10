import openai
import requests
import json

import Tinder.tinder_api as tinder_api
import Tinder.config as config


def openai_response(prompt):
    openai.api_key = config.openai_key
    response = openai.Completion.create(engine="davinci", prompt=prompt, max_tokens=20, temperature=0.9, n=1, stop='\n')

    text = response.get('choices')[0].get('text')

    return text

def spam_likes():
    users = tinder_api.get_recommendations();
    ids = [u.get('_id') for u in users.get('results')]
    # make matches
    for u_id in ids:
        # match em all
        tinder_api.like(u_id)

def send_messages():
     # Set up the matches
   

    matches = tinder_api.all_matches(20).get('data').get('matches')

    personal_id = '5439f354535c7be51b3521b4'

    "Sample message:"
    """
    Rob is a 24-yr old male, online dating and chatting with a female on Tinder and trying to get a phone number. He's pretty funny.\n\n
    """

    for match in matches:
        m_id = match.get('id')
        messages = match.get('messages')
        hers = []
        his = []
        for message in messages: 
            if message.get('to') == personal_id:
                hers.append(message.get('message'))
            else:
                his.append(message.get('message'))
        bio = match.get('person').get('bio')
        name = match.get('person').get('name')

        # avoid last message

        if len(hers) == 0 and len(his) == 0:
            tinder_api.send_msg(m_id, "You're unbelievably sexy.")
        elif len(hers) > 0 and len(his) > 0:
            response = openai_response(f"Rob is a 24-yr old male, online dating and chatting with {name} on Tinder and trying to get a phone number. He's pretty funny.\nRob: {his[0]}\n{name}: {hers[0]}\nRob:")
            tinder_api.send_msg(m_id, response)
        elif len(hers) > 0: 
            response = openai_response(f"Rob is a 24-yr old male, online dating and chatting with {name} on Tinder and trying to get a phone number. He's pretty funny.\n{name}: {hers[0]}\nRob:")
            tinder_api.send_msg(m_id, response)

def main():
    # make call to tinder api
    # Referenced with: https://github.com/fbessez/Tinder

    # Authorize the file
    tinder_api.authverif();
    send_messages()
    spam_likes()

   

if __name__ == "__main__":
    main()