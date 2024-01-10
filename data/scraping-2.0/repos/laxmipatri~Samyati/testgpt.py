# packages you need
import os
import openai
import json


# API key from open AI
openai.api_key = ''

def show_my_itirenary():   
  #The inputs we are feeding to chat gpt will change when we have the htttp request
  startloc = input('Start Location: ')
  print(startloc)
  endloc = input('Destination: ')
  print(endloc)
  loc = input ('One more location')
  print(loc)
  start_dt = input('Start date: ')
  print(start_dt)
  end_dt = input('End date: ')
  print(end_dt)
  budget = input('Budget: ')
  print(budget)
  interests = input('Your interest: ')
  print(interests)


  # Chat entry to chat gpt
  # Model is the api
  # message is an f string where we are injecting the inputs from above
  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "user",
      "content": f"Display response in a JSON for a road trip from {startloc} to {endloc}. Find me Marriott hotels and nearby tourist attractions if my trip is from {start_dt} to {end_dt} for a budget of {budget}. If my preferences include {interests}. Please make itinerary which includes Marriott properties between the source and destinaiton and conrresponding near by tourist attractions/activities and their latitude and longitude and as per weather please suggest any required gear."}
    ]
  )

  # print the response from gpt3
  print(completion.choices[0].message.content)
  return completion.choices[0].message.content
  #content = json.loads(completion.choices[0].message.content)
  # Write the JSON content to a new file
  # with open('data.json', 'w') as file:
  #     json.dump(content, file)

# newInterests = input('Your updated interests: ')
# print(newInterests)

# completion2 = openai.ChatCompletion.create(
#   model="gpt-3.5-turbo",
#   messages=[
#     {"role": "user",
#     "content": f"Display response in a JSON for the same road trip can find me Marriott hotels and some other tourist attractions with same budget, if my preferences include {newInterests}. Please make itinerary which includes Marriott properties between the source and destinaiton and conrresponding near by tourist attractions/activities and their latitude and longitude and as per weather please suggest any required gear."}
#   ]
# )

# # print the 2nd response from gpt3
# print(completion2.choices[0].message.content)

