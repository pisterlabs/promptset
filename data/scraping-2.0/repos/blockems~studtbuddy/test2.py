import openai
import os
from dotenv import load_dotenv
import json

load_dotenv()  # load environment variables from .env file

openai.api_key = os.getenv('OPENAI_API_KEY')  # get OpenAI key from environment variables
port = os.getenv('getSkills_PORT')
debug = os.getenv('getSkills_DEBUG')

# we will use GPT-3 for most of the examples in this tutorial
#guidance.llm = guidance.llms.OpenAI("text-davinci-003")

role = "Agile Coach"
competency = "Emergent"
jsonformat = {
        "Role": "Role Name",
        "Level": "Competency Level",
        "Description": "Role Description",
        "Required": [
              {
                  "Name": "Element Name",
                  "Level": "Element Level",
                  "Description": "Element Description",
                  "Skills": [
                      "Skill 1",
                      "Skill 2",
                      ...
                  ]
              },
              ...
          ],
        "Recommended": [
              {
                  "Name": "Element Name",
                  "Level": "Element Level",
                  "Description": "Element Description",
                  "Skills": [
                      "Skill 1",
                      "Skill 2",
                      ...
                  ]
              },
              ...
          ]
      }

def get_skills():

    prompttext = f'''
    As an expert HR consultant, I am evaluating the competencies necessary for the role of "{role}" at the "{competency}" level, using a competency scale of emergent, competent, expert, and lead. 

    Please generate a comprehensive overview that includes skills, knowledge, key deliverables, and personal attributes contributing to success in this role. 

    Present the information as a JSON object with the following structure:
    '''
    prompttext = prompttext + str(jsonformat)

    prompttext = prompttext + '''
    Within the required and recommended array, each object should represent a key competency required for the role. 
    Each competency should be broken down into sub-elements. 
                  
    Ensure the "Skills" array contains specific skills associated with each competency.''' 

    print(prompttext)

    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo-16k",
      messages=[
        {
            "role": "assistant",
            "content": prompttext
        }
      ]
    )

    if response['choices'] and response['choices'][0]['message']:
      return response['choices'][0]['message']['content'].strip()
    
def get_review(data):
    prompttext = f'''You are an expert {role}. Reviewing the skills for You and '''
    return prompttext

if __name__ == '__main__':
    firstpass = get_skills()
    print(firstpass)
    get_review(firstpass)