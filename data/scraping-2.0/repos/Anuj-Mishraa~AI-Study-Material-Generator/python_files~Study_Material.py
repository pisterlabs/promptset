'''
Created by Anuj Mishra 21/08/2023

I this peice of code the actually AI call is done with openai gpt-3.5-turbo for creating a study material 
On this we can see that we have created the prompt to generate a json data for the study materia and then this json data is further converted into pdf
by the pdf file that is imported here and ahving makepdf function that is called in the last.

All the tests for the process that has been done can beseen in the json_files directory that contaons the json file 
and also the pdfs in the pdf_files
'''

import openai
import json
import images
import pdf
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain import LLMChain
import lang
# Set your OpenAI API key here
openai.api_key = lang.API_KEY
# Define the input prompt with placeholders for title and points
#     api_key = "customize_google_search_engine_api_key"
#     cx = "your_customize_google_search_engine_id"
def lgchain(text, language):
    llm = OpenAI(model_name="gpt-3.5-turbo")
    template =""" translate the following text into {} language
    {}
    """
    prompt = PromptTemplate(template=template,input_variables=[language,text])
    chain = LLMChain(llm=llm,prompt=prompt)
    return chain.run(input)
# Provide the title for the study material
title = input("Enter the topic for your study")
# Define a list of points you want to include
points_list = [
    "Introduction",
    "History",
    "Working Principle",
    "Proof of Law",
    "Adwantages",
    "Disadvantages"
    "Conclusion"
]
# Convert the list of points into a formatted string with image tags
formatted_points = "\n".join(points_list)
a = ["Introduction","working Principal","law"]
urls = "\n".join([images.get_image_url(f"{i} of {title}", api_key, cx)[1] for i in a])
# Combine the input prompt with the title and points using .format
prompt =  """
Create a Study Material on the following topic
{}
Do make the Study Material based upon the following Points
{}
As the response you have to generate a json script each point should be as a json tag and add one img tag intialize withe none after each point tag. consider the following example
Note: Json syntax must be accurate and the encoding must be utf-8
{{
  "Title": "Photosynthesis",
  "Introduction": {{
    "point": "Photosynthesis is a vital biological process that occurs in plants, algae, and some bacteria. It involves the conversion of light energy into chemical energy in the form of glucose and other organic compounds. This process is responsible for producing oxygen, which is essential for supporting life on Earth.",
    "img": null
  }},
  "History": {{
    "point": "The process of photosynthesis was first studied by Jan Ingenhousz in the late 18th century. He discovered that plants release oxygen during the day in the presence of light. In the early 19th century, researchers like Julius von Sachs and Theodor Engelmann further advanced the understanding of photosynthesis. However, it was not until the mid-20th century that the complete mechanism of photosynthesis was elucidated.",
    "img": null
  }},
  "Working Principle": {{
    "point": "Photosynthesis occurs mainly in the chloroplasts of plant cells. The process can be divided into two stages: the light-dependent reactions and the light-independent reactions (Calvin cycle). In the light-dependent reactions, light energy is absorbed by chlorophyll and other pigments, leading to the generation of ATP and NADPH. These energy carriers are then used in the Calvin cycle to convert carbon dioxide into glucose.",
    "img": null
  }},
  "Proof of Law": {{
    "point": "Photosynthesis follows several fundamental principles:\\n\\n1. The Law of Conservation of Energy applies to photosynthesis. The total energy in the system before and after the process remains constant.\\n2. The rate of photosynthesis is influenced by factors like light intensity, carbon dioxide concentration, and temperature.\\n3. The overall equation for photosynthesis is: 6 CO2 + 6 H2O + light energy → C6H12O6 + 6 O2",
    "img": null
  }},
  "Advantages": {{
    "point": "Photosynthesis plays a critical role in maintaining the Earth's ecosystem. It produces oxygen, which is essential for respiration and the survival of many organisms. Additionally, photosynthesis is the foundation of the food chain, as it provides energy-rich organic molecules that sustain various life forms.",
    "img": null
  }},
  "Disadvantages": {{
    "point": "While photosynthesis is crucial, it has limitations. The process can be affected by factors such as limited light availability, water scarcity, and environmental stress. In some cases, excessive exposure to light can lead to photooxidative damage, disrupting the balance of the photosynthetic machinery.",
    "img": null
  }},
  "Conclusion": {{
    "point": "Photosynthesis is a fundamental process that drives the biosphere by converting light energy into chemical energy. It has a profound impact on the global environment, shaping ecosystems and supporting life. Understanding the mechanisms of photosynthesis can lead to advancements in agriculture, renewable energy, and environmental conservation.",
    "img": null
  }}
}}

add the following links to img tags of Introduction, Working Principal and proof of law
{}
""".format(title,formatted_points,urls)
print(prompt)
# Generate content based on the prompt using GPT-3.5-turbo model
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that generates study materials in a auurate json format."},
        {"role": "user", "content": prompt}
    ]
)
# Print the generated content
json_data = response.choices[-1].message["content"]
with open(f'{title}.json', 'w',encoding='utf-8') as outfile:
    outfile.write(json_data)
print(json_data)
json_data = json_data.encode('utf-8')  # Encode the string as utf-8
data = json.loads(json_data)
pdf.makepdf(data,title)
