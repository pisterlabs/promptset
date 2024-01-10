import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
from imessage_reader import fetch_data
import pprint
import json
from openai import OpenAI
# from config import api_key
from dotenv import load_dotenv
import os

#command to pull variables from .env file
load_dotenv()

API_KEY = os.getenv("openai_api_key")





# TODO: read this https://blog.langchain.dev/tutorial-chatgpt-over-your-data/
# 1. put message data into vector embeddings
# 2. idea: when a user clicks on a node, it will expand to show you sentiment, suggestions of more convo, and other details
# 3. We still need to figure out how to get users response to text messages.
# 4. Scope creep : allow the use to input their own promt to the openai api
# 5. I could label the nodes with phone number and number of messages sent 
# 6. I want node size to correspond to the number of messages sent
# 7. I want the node color to correspond to the number of messages sent



client = OpenAI(api_key=API_KEY)


# # Define the command to execute the JavaScript script
# command = ['node', 'number_handler.js']
# # # Execute the command and capture the output
# output = subprocess.check_output(command, universal_newlines=True)
# print("output: ", output)

# Create a FetchData instance
DB_PATH = os.getenv("DB_PATH")
fd = fetch_data.FetchData(DB_PATH)
#this function adds a database to the project directory
fetch_data.FetchData.show_user_txt(fd, "sqlite")

# Store messages in my_data
# This is a list of tuples containing user id, message and service (iMessage or SMS).
message_data = fd.get_messages()


# pprint.pprint(message_data)
# print(message_data[0][1], '\n', message_data[0][2],'\n', message_data[0][0],'\n')

# lets make a dictionary of the messages. The keys will be the user id and the values will be a list of the messages.
# the messages will be important to analyze the sentiment of the messages and other things.

# TODO: 
# 1. put the logic in a function named get_messages

# Create a dictionary to store the messages
def get_contact_count(message_dict):
    return len(message_dict.keys())

message_dict = {}
message_count ={}

for message in message_data:
    if len(message[0])<10:
        continue
    # the replace function is used to remove the leading +1 from the phone number credit to copilot
    # I need to make the replace more generic to handle other country codes
    phone_nubmer = message[0].replace('+1', '')
    # print("test x")

    if phone_nubmer not in message_dict:
        message_dict[phone_nubmer] = [message[1]]
        message_count[phone_nubmer] = 1
    else:
        message_dict[phone_nubmer].append(message[1])
        message_count[phone_nubmer] += 1




#TODO: subsitiute the hard coded conversation with the message_dict  of a specific contact
# Example conversation data

#eventuall I want to be able to use the contacts name instead of the phone number
while True:
    desired_contact = input("\nEnter the phone number of the contact you want to analyze: ")

    if desired_contact in message_dict:
        print("contact",desired_contact, "found! ")
        print("Starting analysis... \n")
        print("contact",desired_contact, "has sent", message_count[desired_contact], "messages")
        print(message_dict[desired_contact])
        conversation = message_dict[desired_contact]
        break
    else:
        print("\ncontact not found! Try again. or press ctrl + c to exit")
        continue


# Joining the conversation into a single string

conversation_text = "\n".join(filter(None, conversation))

# Constructing the prompt for sentiment analysis
content = f"""Analyze the sentiment of a list of text messages, already loaded into a Python list, 
using an AI-based sentiment analysis tool. analyze the content,and determine if the sentiment is positive, 
negative, or neutral. Then, examine the content to identify key themes, subjects, or recurring patterns. 
Based on this analysis, generate insightful talking points or suggestions that can either delve deeper into the 
prevalent topics or introduce new, related subjects. These talking points should be contextually relevant and aim to 
enrich the conversation.keep your response as concise as possible. give me bullet points:\n{conversation_text}"""

# Sending the request to OpenAI API
response = client.completions.create(model="text-davinci-002", prompt=content,
        max_tokens=2000,  # Extended for longer responses
        temperature=0.5,  # Adjust for creativity
        top_p=1,  # Control response diversity
        frequency_penalty=0,  # Fine-tune word frequency
        presence_penalty=0  # Fine-tune word presence
    )

# Extracting and printing the response
print("GPT RESPONSE: ",response.choices[0].text.strip())

print(get_contact_count(message_dict))

#I wonder if the value being appended to contacts can be an object that contains the number of messages sent in order to make the node size correspond to the number of messages sent
contacts=[]
for key in message_count:
    contacts.append(key)

G2 = nx.Graph()
G2.add_nodes_from(contacts)

nx.draw(
    G2, with_labels=True, 
    node_color = "blue", 
    font_size = 7, 
    node_size = 100, 
    edge_color = "green", 
    width = 2.0, 
    alpha = 0.5
    )

plt.show(block=False)
wait = input("PRESS ENTER TO CONTINUE.\n")
print("done")









# # Draw graph
# G = nx.Graph()
# G.add_edges_from(friends)
# nx.draw(G, with_labels=True, node_color = "blue", font_size = 10, 
#         node_size = 100, edge_color = "green", width = 2.0, alpha = 0.5)
# plt.show()


 
#print(nx.nodes(G))

# G = nx.Graph()
# friends = []

# G.add_edges_from(friends)
# nx.draw(G, with_labels=True, node_color = "blue", font_size = 8, bbox=dict(facecolor='red', alpha=0.5), 
#         node_size = 100, edge_color = "green", width = 2.0, style = "dashed", alpha = 0.5)
# plt.show()



# pprint.pprint(message_dict)
#  message dict example
#          } 
#               'phone_number':  ['How fa my bro ',
#                               'Hello brother ? ',
#                               'Are you there my bro ? ',
#                               'Umm you have zelle ? I want to pick $300',
#                               'You can save my iMessage ',
#                               'Are you there bruh ? ',
#                               'Nah my account was banned',
#                               'Oh ok my bro ',
#                               'How fa my bro ',
#                               'I’m just chilling man',
#                               'What’s up',
#                               'Doing good fam ',
#                               ]
#            }    