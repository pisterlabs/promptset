from twilio.rest import Client 
from metaphor_python import Metaphor 
import openai 
import googletrans 
from datetime import datetime, timezone
from bs4 import BeautifulSoup 
import random

translator = googletrans.Translator() 
OPENAI_KEY = "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"
METAPHOR_TOKEN = "143f097d-ed93-47d7-a022-8f87a3d28f2b"
SID = "AC29ae5a39b7d6bbd6244e22532fb87425"
AUTH_TOKEN = "22c544f71bbb6449c044eaa5744de188"
NUMBER = '+18559593981'

openai.api_key = OPENAI_KEY 
metaphor = Metaphor(METAPHOR_TOKEN) 
account_sid = SID 
auth_token = AUTH_TOKEN 

client = Client(account_sid, auth_token) 

def create_weekly_info():
    # We get a list of ranks by calling metaphor to classify which answer was the best 
    USER_QUESTION = "Easy to read and simple Articles on Financial Literacy. The information should be very easy to understand. You should always assume the user is reading the information from a smartphone"  
    SYSTEM_MESSAGE = "You are a helpful assistant that generates search queiries based on user questions. Generate a search query that pertains to common banking terms and banking trivia. Only generate one search query."

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": USER_QUESTION},
        ],
    )
    query = completion.choices[0].message.content
    search_response = metaphor.search(
        query, use_autoprompt=True
    )
    
    # Chose a random url in search response and ask a question based on that
    
    content = search_response.get_contents()
    random.shuffle(list(content.contents))
    first_five_results = content.contents[:5]
    # print()
    for i in range(len(first_five_results)):
        first_five_results[i] = BeautifulSoup(first_five_results[i].extract).get_text()
    
    # print(cont)
    SYSTEM_MESSAGE = "You are a helpful assistant that summarizes financial information into content that even a 10 year old can understand. You will be given a ton of information, and you should write 1 memorizable paragraph on content in financial literacy."

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": "\n".join(first_five_results)},
        ],
    )    
    # print(first_five_results)
    question = completion.choices[0].message.content

    # mapped_variables = {
    #     "question": question,
    #     "true_answer": ,
    #     "completion": set((completion.choices[0].message.content).split(';'))
    # }
    
    return question 
    # return mapped_variables 
def save_weekly_information(information):
    #use pickle to save this information# }
    import pickle 
    with open("weekly.pickle", 'wb') as file: 
      pickle.dump(information, file)    

def load_weekly_information(information):
    import pickle 
    content = ""
    with open("weekly.pickle", 'rb') as file:
        content = pickle.load(file)
    return content 

    
   
if __name__ == "__main__":
    create_weekly_info()