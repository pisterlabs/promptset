#####################################################################################################################
#                                                                                                                   #
# User should be able to enter a prompt and system will search for corresponding URL                                #
# GPT/Gemini will search correct URL, open a browser and take a screenshot out of it.                                      #
# Based on the screenshot, we'll use vision APIs to understand the screenshot, extract content out of it and        #
# give the result back to user by extracting text out of it.                                                        #
#                                                                                                                   #
# We'll use puppeteer in node.js to open URL and take screenshot.                                                   #    
# We'll use GPT4/Gemini to get the URLs                                                                                    #
# We'll use GPT4-vision/Gemini-Vision API to understand the images.                                                               #
#                                                                                                                   #
# We'll have user give option to use GPT or Gemini                                                                  #
#                                                                                                                   #
# Inspiration: https://www.youtube.com/watch?v=VeQR17k7fiU - Easy Web Scraping                                      #                                                                             #
#####################################################################################################################     


import openailib
import googlelib
import utilities
from dotenv import load_dotenv
load_dotenv()

llm_model = input('Choose the LLM to use, 0 for OpenAI GPT and 1 for Google Gemini: ')

if(llm_model !=  '0' and llm_model != '1'):
    llm_model = '0' #Defaults to GPT 

question = "What is the cost of cricket Bat in India?"

if(llm_model == '1'):
    print('You are using Google Gemini LLM ')
    #Pass in the question to gemini and get the url for the answer
    url = googlelib.call_google_chat(question)
else:
    print('You are using OpenaI GPT LLM ')
    #Pass in the question to gpt and get the url for the answer
    url = openailib.call_openai_chat(question)

#run puppeteer to take a screenshot of the url and save it as screenshot.jpg
utilities.run_node_subprocess("puppeteer.js", url)

# Function to encode the image
base64_image = utilities.encode_image("screenshot.jpg")

if(llm_model == '1'):
    #call gemini vision api to read from the image and get the answer to the quesiton
    image_response = googlelib.call_google_vision(question, "screenshot.jpg")
else:
    #call gpt vision api to read from the image and get the answer to the quesiton
    image_response = openailib.call_openai_vision(question, base64_image)

print(image_response)