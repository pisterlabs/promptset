import openai
import dotenv
import os

import re
# from chat.chatbot import chatbot
from gmaps.connect_gmap import get_restaurant_recommendations, add_to_coordinate
from typing import List, Dict
from chat.utils import filter_text

dotenv.load_dotenv()
openai.api_key = os.environ.get("API_KEY")


# def extract_coordinate(recom, text):
#     address_pattern = r"(\d+\s+[\w\s]+,\s+[\w\s]+,\s+\w+)"
#     name_pattern = r'@([^@]+)@'
#     names = []
#     addresses = []
#     matches = re.findall(address_pattern, text)
#     for match in matches:
#         addresses.append(match)
#     matches = re.findall(name_pattern, text)
#     for match in matches:
#         names.append(match)

#     return [[names[idx], addresses[idx], add_to_coordinate(addresses[idx]), recom[names[idx]]['photos']] for idx in list(range(min(len(names), len(addresses))))]

def extract_placeid(text):
    pattern = r"[A-Z][a-zA-Z0-9_\-]{26}"
    matches = re.findall(pattern, text)
    return matches

def extract_coord(recom, text):
    ID = extract_placeid(text)
    print("total recommendation number:", len(ID))
    print("they are:", ID)
    return [[recom[i]['name'], recom[i]['address'], add_to_coordinate(recom[i]['address']), recom[i]['photos'], recom[i]['reviews'], recom[i]['url']] for i in ID]


# def extract_text_between_at_signs(text):
#     pattern = r'@([^@]+)@'
#     matches = re.findall(pattern, text)
#     return matches

def get_top5(messages_history, recom, model="gpt-3.5-turbo", max_tokens=300):
    # TODO reviews
    modified_recs = [recom[r]['name'] + " (" + f"rating: {recom[r]['rating']}" + ", " + f"distance: {recom[r]['distance']}" 
                     + ", " + f"ID: {r}" + ")" for r in recom.keys()]
    findmsg = f"Choose AT MOST 3 best restaurants from the list provided and give me all of their 27-digit ID: {modified_recs}."

    messages_history += [{"role": "user", "content": findmsg}]
    ridmsg = "Give me a list of the NAME of each of the restaurant (there should be only 3 names in this list) in bullet points"

    output = openai.ChatCompletion.create(
        model=model,
        messages=filter_text(messages_history),
        max_tokens=max_tokens
        )

    output_text1 = output['choices'][0]['message']['content']
    messages_history += [{"role": "assistant", "content": output_text1}]

    messages_history += [{"role": "user", "content": ridmsg}]
    output = openai.ChatCompletion.create(
        model=model,
        messages=filter_text(messages_history),
        max_tokens=max_tokens
        )
    output_text2 = output['choices'][0]['message']['content']
    messages_history += [{"role": "assistant", "content": output_text2}]

    messages_history.pop(-2)
    messages_history.pop(-2)
    messages_history.pop(-2)

    return output_text1, extract_coord(recom, output_text1)


def chatbot(messages_history: List[Dict[str, str]], model="gpt-3.5-turbo", max_tokens=400):
    initial_message = {
        "role": "user",
        "content": "You are a helpful assistant designed to help me choose a restaurant. You should ask a series of questions to learn my preferences so that you can suggest a tailored food place recommendation based on my preferences. Go ahead and introduce yourself as a helpful AI assistant designed to help me choose a restaurant and get started with asking the first question to determine my preferences.",
    }

    
    messages_history.insert(0, initial_message)

    output = openai.ChatCompletion.create(
        model=model,
        messages=messages_history,
        max_tokens=max_tokens
        )
    output_text = output['choices'][0]['message']['content']
    messages_history += [{"role": "assistant", "content": output_text}]
    messages_history.pop(0)

    return messages_history

# mes_hist = chatbot([])
# mes_hist.append({"role": "user", "content": "we want to get meat close to Berkeley campus"})
# mes_hist = chatbot(mes_hist)

# reco = get_restaurant_recommendations("meat food", (-122.257740, 37.868710))

# output_text = get_top5(mes_hist, reco)
# print(output_text)

def converter(mes_hist, reco, user_location):
    text, output = get_top5(mes_hist, reco)
    print(text)
    ret = []
    for i in range(len(output)):
        ret.append({"role": "assistant", "content": output[i][0] + " (" + output[i][1] + ")"}) #name, address
            
        if len(output[i][3]) == 0:
            continue
        elif len(output[i][3]) == 1:
            ret.append({"role": "assistant-image", "content": output[i][3]}) #link
        else:
            ret.append({"role": "assistant-images", "content": output[i][3]}) #link
        ret.append({"role": "assistant", "content": output[i][4]}) #reviews

    markers = {}
    markers[len(output)] = {'lat': user_location['lat'], 'lng': user_location['lng'], 'title': "Your Location"}
    for i in range(len(output)):
        markers[i] = {'lat': output[i][2]['lat'], 'lng': output[i][2]['lng'], 'title': output[i][0], 'link': output[i][5]}

    ret.append({"role": "assistant-map", 
                    "content": {'center': user_location,
                                'zoom': 13,
                                'markers': [markers[i] for i in range(len(markers))]
                                }
                    })
    return ret

# print(converter({'lat': 'lalala', 'lng': 'lololo'}))