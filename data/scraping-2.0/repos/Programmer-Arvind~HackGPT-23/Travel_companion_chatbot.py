# It is the basic CLI version of Travel Companion
# HackGPT project

import openai

# Using api key saved in the text file
openai.api_key_path = "./hackGPT_api_key.txt"


# User inputs
destination = input("Enter the state you want to travel to : ")
start = input("Enter your homestate : ")

# Getting output from gpt-3.5-turbo model
message = [{"role" : "user", "content" : f"""I want to travel to {destination} from {start}.
            Give me 6 common phrases used there in day to day life, which is the best travel medium ,
            the tourist places, famous food items in eighty words. Give it in bullets for everything. 
            Give me a python list named food with the food items mentioned above (dont do anything with it)"""} ]
completion = openai.ChatCompletion.create(
    model = "gpt-3.5-turbo",
    messages = message
)
reply = completion.choices[0].message.content

# Printing the reply
print(reply)
