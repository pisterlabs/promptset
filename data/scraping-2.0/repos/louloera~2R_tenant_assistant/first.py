import os 
import openai
import requests
# from .env import GPT_TOKEN



# openai.organization = ""
# openai.api_key= os.getenv("GPT_TOKEN")
openai.api_key= 'PUT KEY HERE'
openai.Model.list()


path = 'http://localhost:5000/home/2'
response = requests.get(path)
response_body = response.json()
address= response_body['address']

path = 'http://localhost:5000/home/2'
response = requests.get(path)
response_body = response.json()
checkout= response_body['checkout']

path = 'http://localhost:5000/home/1/trash'
response = requests.get(path)
response_body = response.json()
days= response_body['days']
time = response_body['time']

path = 'http://localhost:5000/home/1/towels'
response = requests.get(path)
response_body = response.json()
towel_location= response_body['location']

path = 'http://localhost:5000/home/1/items'
response = requests.get(path)
response_body = response.json()
item_name= response_body['name']
item_location = response_body['location']




messages = [
    {"role": "system", "content": f'You are an assistant for a short term tenant in a home in {address}. They need your help to answer any questions related to\
      the surrounding areas. You know information about the house that can help them during their stay. House infromation: The check out is at {checkout}:00pm.\
     The trash is picked up on {days} at {time}:00. Extra towels are located at {towel_location}. The {item_name} is located at {item_location}'}
]
while True: 
    message = input("User: ")
    if message:
        messages.append(
            {"role":"user", "content":message},
        )

        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
    reply = chat.choices[0].message.content
    print(f"CHATGPT: {reply}")
    messages.append({"role":"assistant", "content":reply})











# def chat_with_chatgpt(prompt, model="gpt-3.5-turbo"):
#     response = openai.ChatCompletion.create(
#         model=model,
#         messages=prompt,
#         # max_tokens=100,
#         n=1,
#         # stop=None,
#         temperature=0.5
#     )
#     message= response['choices'][0]['message']['content']
#     return message
    
# user_prompt = "Write a summary of the benefits of coding"
# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Tell me a joke."}
# ]
# chatbot_response = chat_with_chatgpt(messages)
# print(chatbot_response)
