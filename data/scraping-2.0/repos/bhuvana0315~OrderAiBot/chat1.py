import openai
import os
from dotenv import load_dotenv
import json
from twilio.rest import Client

load_dotenv()

api_key = os.getenv("API_KEY")
openai.api_key = api_key

# Rest of your code using the OpenAI library

context = [ {'role':'system', 'content':"""
You are OrderBot, an automated service to collect orders for a street dosa. \
You first greet the customer as Hello I am an orderbot,Your first question after greeting the customer how may I help you today.This question is first question and fixed\
then collects the order, \
and then asks if it's a pickup or delivery. \
You wait to collect the entire order, then summarize it and check for a final, all amount are in Rupees \
time if the customer wants to add anything else. \
Make sure to clarify all options, extras and sizes uniquely \
identify the item from the menu.\
If it's a delivery, you ask for an address. \
Finally you collect the payment for all the orders.\
You should respond only to take the orders and for all other questions you should not respond since you are an orderbot. \
You respond in a short, very conversational friendly style. \
You should take orders only for the items that aree included in the following menu. \
The menu includes \

Plain dosa: 60/-
Onion dosa: 70/-
Onion butter dosa: 70/-
Ghee dosa: 70/-
Karam dosa: 70/-
Avakaya dosa: 70/-
Paneer dosa: 80/-
Ghee karam dosa: 80/-
70mm dosa: 110/-
Ghee karam 70mm dosa: 110/-
Masala Dosa: 70/-
Onion Masala Dosa: 80/-
Butter Masala Dosa: 90/-
Paneer Masala Dosa: 90/-
Avakaya Masala Dosa: 90/-
Cheese Masala Dosa: 100/-
Ghee Karam Masala Dosa: 100/-
Schezwan Dosa: 180/-
Butter Schezwan Dosa: 180/-
Paneer Schezwan Dosa: 180/-
Cheese Schezwan Dosa: 180/-
Ravva Dosa: 60/-
Onion Ravva Dosa: 70/-
Ravva Masala Dosa: 70/-
Ghee Ravva Dosa: 70/-
Onion Ravva Masala Dosa: 80/-
Cheese Ravva Dosa: 80/-
Cheese Ravva Masala Dosa: 90/-
Ravva Upma Dosa: 90/-
Onion Ghee Dosa: 80/-
Butter Dosa: 70/-
Steam Dosa: 90/-
Set Dosa: 100/-
Uthappam: 60/-
Onion Uthappam: 70/-
Mixed Uthappam: 90/-
Pesaraptu: 60/-
Onion Pesaraptu: 70/-
Mixed Pesaraptu: 90/-
Ghee Mila Pesarattu: 110/-
Ghee Mila Dosa: 110/-
Ghee Paneer Dosa: 130/-
Cheese Paneer Dosa: 130/-
Butter Paneer Dosa: 130/-

"""} ]  # accumulate messages

def get_completion_from_messages1(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
#   print(str(response.choices[0].message))
    return response.choices[0].message["content"]

def collect_messages_text1(msg):
    prompt = msg #.split(' ')
    print(prompt)
    # word='pickup'
    # if(word in prompt): 
    if(prompt=="pickup" or prompt=="delivery"):
        store_order_summary()
    context.append({'role':'user', 'content':f"{prompt}"})
    response = get_completion_from_messages1(context) 
    context.append({'role':'assistant', 'content':f"{response}"})
    return response


def store_order_summary():
    context.append({'role':'user','content':'Store the order in a json format with fields containing items,quantity and total price'})
    response = get_completion_from_messages(context) 
    # context.append({'role':'assistant', 'content':f"{response}"})
    print(response)
    with open('order_summary.json', 'w') as json_file:
        json.dump(response, json_file)
    user_phone_number='+919347665827'
    send_whatsapp_message(user_phone_number, response)

def send_whatsapp_message(to, body):
    # Twilio credentials
    twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    twilio_whatsapp_number = os.getenv("TWILIO_WHATSAPP_NUMBER") # Twilio's sandbox WhatsApp number

    client = Client(twilio_account_sid, twilio_auth_token)

    try:
        message = client.messages.create(
            from_=twilio_whatsapp_number,
            to='whatsapp:' + to,
            body=body
        )

        print('WhatsApp message sent successfully.')
        print(message.sid)
    except Exception as e:
        print('Error sending WhatsApp message:', str(e))
