

import openai
import readchar 
import streamlit as st

mykey = st.secrets["API_KEY"]
openai.api_key = mykey

keepGoing = True
while keepGoing:
  textToCorrect = input("Enter a dish: ")
  response = openai.Completion.create(
    model="text-davinci-003",
    prompt=f"Give me a step by step recipe for: {textToCorrect}. List all the ingredients at the top",
    temperature=0,
    max_tokens=1000,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
  )

  print(response["choices"][0]["text"])
  print()

  print("Press y to continue, n to quit")
  keepGoing = readchar.readkey() == "y"