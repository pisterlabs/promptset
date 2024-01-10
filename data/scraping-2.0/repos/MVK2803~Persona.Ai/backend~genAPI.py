import openai
import json
import os
from dotenv import load_dotenv
def generatePersona():
   load_dotenv()


   openai.api_key = os.getenv('OPENAI_API_KEY')

   
   prompt = """
   Generate a random persona in JSON format for a fictional individual. Include the following details:

   1. **Name**: [Random Name]
   2. **Job**: [Random Job Title]
   3. **Quote**: [A brief quote that represents the persona's outlook]
   4. **Age**: [Random Age]
   5. **mStatus**: [Married/Single/Other]
   6. **Location**: [Random Location]
   7. **About**: [A short description of the persona's background and interests]
   8. **Needs and Goals (in points)**:
      - [Specific need or goal 1]
      - [Specific need or goal 2]
      - [Specific need or goal 3]
      

   9. **Pain Points**: 
      - [Pain point 1]
      - [Pain point 2]
      - [Pain point 3]

   10. **Attitude and Behavior (in points)**:
      - [Positive attitude or behavior 1]
      - [Positive attitude or behavior 2]
      - [Negative attitude or behavior 1]
      - [Negative attitude or behavior 2]

      
   11. **Personality**:
      [% of introvertness, % of insecurity, % of pssiveness] just percentages in a list

      

   12. **Motivation**:
      - [PLeasure in %] just percentage dont give the title
      - [Validation in %] just percentage dont give the title
      - [Health in %]   just percentage dont give the title
      - [Fear in %]   just percentage dont give the title

   """

   
   response = openai.Completion.create(
   engine="text-davinci-003",  
   prompt=prompt,
   temperature=0.7,
   max_tokens=500
   )

   # Extract the generated text from the API response
   generated_text = response['choices'][0]['text']
   

   
   start=generated_text.index('{')    # Remove the backticks from the generated text
   end=len(generated_text)-(generated_text[::-1].index('}'))
   generated_text=generated_text[start:end]
   persona_json = json.loads(generated_text)
   result=['Name', 'Job', 'Quote', 'Age', 'Marital', 'Location', 'About', 'Goals', 'Painpts', 'Attitude', 'Personality', 'Motivation']
   resultJson={}
   for index, (key, value) in enumerate(persona_json.items()):
      resultJson[result[index]]=value
   prompt_img= "Photo for  {}year old {} from {} doing {}".format(resultJson["Age"],resultJson["Name"],resultJson["Location"],resultJson["Job"])
   response = openai.Image.create(
        prompt=prompt_img,
        n=1,
        size="256x256",
    )
   resultJson['Image']=response['data'][0].url
   return(resultJson)
   # Convert the generated text to a JSON object (assuming it follows a JSON-like structure)

