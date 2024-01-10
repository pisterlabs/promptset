#!/usr/bin/env python
import cv2
from pyzbar import pyzbar
from pymongo import MongoClient
from bson.objectid import ObjectId
import os
import openai

customer_data = None
def decode_qr_code(frame):
    decoded_objects = pyzbar.decode(frame)
    for obj in decoded_objects:
        qr_code_data = obj.data.decode()
        print("QR Code Detected:", qr_code_data)
        print(type(qr_code_data))
        return qr_code_data
    return None

def fetch_customer_details(customer_id):
    mongoUri = 'mongodb+srv://chandran0303cn:POXVa3cpoagtpU7q@cluster0.iq6t67w.mongodb.net/?retryWrites=true&w=majority'
    client = MongoClient(mongoUri)
    db = client['Supermarket_User_Profile']
    collection = db['User_preferences']

    try:
        document = collection.find_one({'_id': ObjectId(customer_id)}, {'contact': 0, 'email': 0})
        return document
    except Exception as e:
        print("Error fetching customer details:", e)
        return None
    finally:
        client.close()

def fetch_store_details():
    mongoUri = 'mongodb+srv://chandran0303cn:POXVa3cpoagtpU7q@cluster0.iq6t67w.mongodb.net/?retryWrites=true&w=majority'
    client = MongoClient(mongoUri)
    db = client['Supermarket_Admin_Environment_Profile']
    collection = db['Supermarket']
    store_details = []
    try:
        for doc in collection.find():
            shelf_no = doc.get('shelf_no', 'Unknown')
            product_name = doc.get('productName', 'Unknown')
            store_details.append([shelf_no, product_name])
    except Exception as e:
        print("Error fetching store details:", e)
    finally:
        client.close()

    return store_details

def chat_bot():
    api_ke = os.environ.get('OPENAI_API_KEY')
    openai.api_key=api_ke
    env_context = fetch_store_details()
    conversation_ended = False
    messages = [
        {"role": "system", "content": "You are a supermarket chatbot who is supposed to help the customer in the store go to the appropriate locations in order to pick up the items of their choosing."},
        {"role": "system", "content": "Firstly, there are 100 shelves in the supermarket and this is the context about the store represented in a list of the following format : [shelf number,type of product it has] - " + str(env_context)},
        {"role": "system", "content": "Secondly, this is the information of the customer you are serving today including their name and information to various parameters to help you tailor your answers based on their preferences -" + str(customer_data)},
        {"role": "system", "content": "Have a conversation with the user until they are ready to begin shopping. When that is detected, end the conversation with 'Destinations - [<comma separated list of all shelves to visit>]'"},
        {"role": "system", "content": "Below you will find some examples for different store contexts and user preferences (this may differ from your own). These examples are meant to make you understand how to respond but using your OWN store context and user preferences given above."},
        {"role": "system", "content": "This is the first example conversation you are having with a customer. This customer has a high degree of price conciousness (4) and wants to shop as fast as possible (4) unless indicated otherwise. There are no dietary restrictions."},
        {"role": "user", "content" : "I want to make standard Dutch pancakes."},
        {"role": "assistant", "content" : "Since you have indicated preferences for high speed shopping and appreciation for trying out new products, may I recommend that you try the pancake mixes on shelf 29? If you would like to make them from scratch however, we can collect all the necessary ingredients as well like eggs, wheat flour, milk and spices like vanilla and cinammon. Please let me know how you would like to proceed."},
        {"role": "user", "content" : "Hmmmm since it is the weekend and I want to make some healthier pancakes, I would prefer making it from scratch."},
        {"role": "assistant", "content" : "I understand. In that case, we will need to visit the following shelves to get the necessary ingredients - shelf 21 for eggs, shelf 40 for flour, shelf 16 for milk and shelf 36 for the spices. Furthermore, since you mentioned protein pancakes, may I recommend adding some Whey protein that you can find on shelf 28?"},
        {"role": "user", "content" : "That sounds like a good idea! I do have milk and flour at home but will need the rest. Also I was thinking of adding some fruits to my pancakes as well."},
        {"role": "assistant", "content" : "That sounds wonderful, then let us not go to the shelves for milk and flour but instead go to shelf 1 and grab you some fresh bananas and blueberries to top up your pancakes!"},
        {"role": "user", "content": "Perfect! Let's go."},
        {"role": "assistant", "content": "Destinations - [shelf21,shelf36,shelf28,shelf1]"},
        {"role": "system", "content": "End of Conversation"},
        {"role": "system", "content": "This is the second example conversation you are having with a different customer who shows a help appreciation (4) and has mentioned in their dietary preferences that they are allergic to nuts and are a vegan."},
        {"role": "user", "content" : "I want to make a healthy bulking protein smoothie"},
        {"role": "assistant", "content" : "Since you had previously indicated that you are a vegan who is also alergic to nuts, may I recommend going to shelf 17 for Pea milk, shelf 28 for plant based protein, shelf 1 for fresh fruits and shelf 27 for oatmeal."},
        {"role": "user", "content" : "That sounds perfect! Also I'll need to clean my place up after a wild party last night. Not sure of what all to get. Can you give a checklist and Ill tell you the stuff I already have?"},
        {"role": "assistant", "content" : "Definitely. A comprehensive checklist would include visiting shelves 62 for household cleaners, 64 for dishwashing supplies like sponges and soap, 66 for trash bags and 65 for towels and rags to clean up the place."},
        {"role": "user", "content" : "I do have a household cleaner but I am not sure of the amount. I also have trash bags at home."},
        {"role": "assistant", "content" : "I see. May I then recommend a small bottle of dishwashing liquid to supplement the amount you already have and visiting all the other shelves except 66 as you have trash bags already?"},
        {"role": "user", "content": "Yes, that sounds ideal. Let's proceed with buying the items."},
        {"role": "assistant", "content": "Destinations - [shelf17,shelf28,shelf1,shelf27,shelf62,shelf64,shelf65]"},
        {"role": "system", "content": "End of Conversation"},
        {"role": "system", "content": "This is the third and final example conversation you are having with yet another different customer. This customer has low price conciousness (1) and has high degree of knowledge (5) of what they need so their requests will be specific and will expect to the point answers."},
        {"role": "user", "content" : "I want to make the authentic Paella from Valencia. I need Paella rice, onions, garlic, spice mix, chicken, bell peppers and some lemon."},
        {"role": "assistant", "content" : "For the ingredients you have listed, let us go to shelf 31 for the rice, shelf 5 for the onions, garlic and peppers, shelf 1 for the lemon, shelf 36 for spices and shelf 10 for the chicken."},
        {"role": "user", "content" : "Sounds good. I would like to pair it with Navarra Rosado as well."},
        {"role": "assistant", "content" : "Definitely. Navarra Roasado and similar wines can be found in shelf 56."},
        {"role": "user", "content" : "I would also like to buy a nice high end dessert though I am not sure of exactly which one. Ideally a good pairing with my dish."},
        {"role": "assistant", "content" : "I would recommend the Crema Catalana to maintain the spirit of an authentic Valencian dinner. The same can be found in shelf 24 along with other desserts."},
        {"role": "user", "content": "Alright. Let us buy all the items now."},
        {"role": "assistant", "content": "Destinations - [shelf31,shelf5,shelf1,shelf36,shelf10,shelf56,shelf24]"},
        {"role": "system", "content": "End of Conversation"}
    ]
    print(messages)
    while not conversation_ended:
        user_input = input("User: ")
        messages.append({"role": "user", "content": user_input})
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages=messages
        )
        assistant_response = response.choices[0].message['content']
        print("Assistant's response:", assistant_response)
        messages.append({"role": "assistant", "content": assistant_response})
        if "Destinations - [" in assistant_response or "EXIT" in user_input:
            conversation_ended = True

    print("Conversation has ended.")



def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if ret:
            decoded_data = decode_qr_code(frame)
            if decoded_data is not None:
                customer_id = decoded_data 
                customer_details = fetch_customer_details(customer_id)
                if customer_details:
                    global customer_data
                    customer_data=customer_details
                else:
                    print("Customer not found or error occurred.")
                break
            cv2.imshow('QR Code Scanner', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
    chat_bot()
if __name__ == '__main__':
    main()

