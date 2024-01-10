import openai
import os
from langchain import PromptTemplate
from langchain.tools import Tool
import requests
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.utilities.dalle_image_generator import DallEAPIWrapper
import tiktoken
import streamlit as st

#Defining all the API keys of the services beings used
os.environ['OPENAI_API_KEY'] = st.secrets["openai_key"]
openai.api_key = st.secrets["openai_key"]
os.environ["SERPER_API_KEY"] = st.secrets["serper_api_key"]

#Defining the prompt that will generate our content primarily
user_template = PromptTemplate.from_template(""" Strictly follow the following instruction while writing the essay : \n 

 (1) THE TOPIC OF THE ESSAY SHOULD BE  *{topic}*, and you should write about it thoroughly.
 (2) THE ESSAY SHOULD HAVE THE FOLLOWING QUALITIES IN ITS STYLE OF WRITING. QUALITIES : *{writing_pattern}.
 (3) IT SHOULD CONTAIN EXACTLY TWO PARAGRAPHS and *NO HEADING*.

 Now return the required essay

 Essay: 
 """
)

#Empty lists will be used to store the images and the list of images
lst_images = []
lst_source_images = []
lst_name_images = []

#Defining the tools we will be using

tools = [
    Tool(
        name="Google Search",
        description="Search Google for recent results.",
        func=GoogleSerperAPIWrapper().run,
    ),
    Tool(
        name="Image Generator",
        description="Generate Image for a query",
        func=DallEAPIWrapper().run,
    ),
    Tool(
        name="Wikipedia Information",
        description="To generate information about a particular topic",
        func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()).run,
    ),
    Tool(
        name="Image Searcher",
        description="To generate images for the generated Context",
        func=GoogleSerperAPIWrapper(type="images").results,
    )

]

#Defining all the functions that are to be used in this code.

#Defining a function to estimate the number of tokens used, since LLMs have limited by the number of tokens we can send them, this needs to be taken care of.
def num_tokens_from_messages(messages):
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(str(messages)))
    return num_tokens

def heading(prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": """Give the following paragraph a heading as found appropriate. \n Paragraph: \n
                     """ + prompt + """ \n.
                      Heading:"""}],

        temperature=0
    )

    heading = completion.choices[0].message.content
    with open("heading.txt", "w") as f:
        f.write(heading)
    return heading

#A function to recognize the pattern and style of content given as input, few shot prompting has been used here to teach the
#model to provide certain kind of responses using two examples.
def writing_pattern(prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": """You are given an input text. Recognize exactly two emotions or qualities from the *tone of the text*. Here are the two examples of the kind of response you are supposed to produce \n
                     Text : "The Headless CMS is the new sexy thing in the software world. Forget about the old lousy stuff" 
                     Emotions: "Very Funny, Very Casual"
                     This is the following text for which you have to generate the emotions. \n
                     Text : "Big O notation is a mathematical notation that describes the limiting behavior of a function when the argument tends towards a particular value or infinity." 
                     Emotions: "Very Technical, Very Informative"
                     This is the following text for which you have to generate the emotions. \n
                     Text : 
                     """ + prompt + """ \n.
                      Emotions:"""}],

        temperature=0
    )

    writing_pattern = completion.choices[0].message.content
    print(writing_pattern)
    return writing_pattern

#Defining the function that will generate the content it uses prompt, context and writing pattern as parameters. Context
#is generated using the tools that have been mentioned above, prompt is the user provided one, and writing pattern is ob
#-tained by using another LLM response. LLM chaining is used over here.
def model_response(prompt, context, heading, writing_pattern, temperature_parameter):
    message_template = prompt_creator(user_input=prompt, writing_pattern=writing_pattern, topic=heading,
                                      context=context)

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message_template,
        temperature=temperature_parameter
    )

    response = str(completion.choices[0].message.content)

    #The model outputs often do not strictly follow instructions so it sometimes give a heading to the text generated. This
    #removes the heading if given.

    for indx in range(len(response)):
        if response[indx]=="\n":
            if indx<115:
                response=response[indx+1:]
            break

    with open("response.txt", "w") as f:
        f.write(response)
    print(response)
    return response

#Now defining the function that will create the final prompt that will be given to out model as input, this prompt will
#consist of the prompt that has been mentioned above along with one shot learning technique included as seen in the user
#messages values below. It is kind of a reward system. We ask the model to generate response about a particular topic
#which we have extracted from the input text, then we say that the response the user provided is exactly the kind that is
#required, thus promoting it to provide more responses of the kind that the user mentioned.
def prompt_creator(user_input, topic, writing_pattern, context):
       print("Beginning Prompt Creation")
       other_template=user_template.format(writing_pattern=writing_pattern, topic=topic)

       system_template = """ You are a bot supposed to *STRICTLY* follow user instructions. Remember you write essays with the following qualities. Qualitites : \n {writing_pattern}"""

       other_templa = system_template.format(writing_pattern=writing_pattern)

       user_message=[{"role":"system","content":other_templa},
                     {"role":"user","content":"Generate some content about the following topic: \n" + topic},
                     {"role": "assistant", "content": user_input},
                     {"role": "user", "content": "Very Good. This is exactly what I was asking for. *now generate response in exactly the same tone as the previous one* and complementing it.\n."+other_template+"You might use the following context incase it is helpful : Context \n "+str(context)},]
       return user_message

def image_generation(images):
    s = 0
    for image in images["images"]:
        lst_source_images.append(image["imageUrl"])
        image1 = requests.get(image["imageUrl"])
        lst_images.append(image1.content)
        lst_name_images.append(image["source"])
        with open("img"+str(s+1)+".jpg", "wb") as f:
            try:
                s = s + 1
                f.write(image1.content)
            except Exception as e:
                print(f"Could'nt load a few elements due to the following errors {e}")

def dallegenerator(image, prompt):
    print(image)
    image = requests.get(image)
    with open("DALLE2"+".jpg", "wb") as f:
        try:
            f.write(image.content)
        except Exception as e:
            print(f"Could'nt load the DALLE image generator image due to the following reasons {e}")
    return image.content
def dalle_prompt(heading):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": """Generate a generic and metaphorical prompt for a image creator app to create the image based on the given input.
            Remeber the image generator does'nt understand technical langauge. So keep the language and the prompt simple and *very specific* as shown in the examples. Here are few examples of Statements and Prompts. \n
            'Statement' : 'Headless CMS is revolutionizing the software world'.
            'Prompt' : 'Visual art of a Headless Guy sitting in front of a Computer and typing'

            'Statement' : 'The power of generative AI in new age content generation'
            'Prompt' : 'Amazing visual art of a Artificial Intelligence Bot writing a letter while sitting on a chair.'

            'Statement' : 'The Rise of web3 and blockchain'.
            'Prompt' : 'Amazing visual art of chains in blocks connected with each other and flying'.

            Similar to the above examples generate a prompt for the following statement.
            'Statement':'
            """+heading+""" '.\n 'Prompt':  """}],

        temperature=0.8
    )

    dalleprompt = completion.choices[0].message.content
    return dalleprompt
def final_model_response(user_prompt,openai_key, serper_api_key, new_heading="-1",temperature_parameter=0):
    os.environ['OPENAI_API_KEY'] = openai_key
    openai.api_key = openai_key
    os.environ["SERPER_API_KEY"] = serper_api_key
    context = " "
    if new_heading=="-1":
        new_heading = heading(user_prompt)
    print(new_heading)

    for tool in tools:
        print(tool.name)
        if tool.name == "Image Generator":
            dalle_prompt_created = dalle_prompt(str(new_heading))
            image = tool.run(dalle_prompt_created)
            dallegenerator(image, dalle_prompt_created)
            continue
        if tool.name == "Image Searcher":
            #This tool is not being used right now.
            continue
        context = context + tool.run(str(new_heading))

    print("Responding")
    return model_response(user_prompt, context, str(new_heading), writing_pattern(user_prompt), temperature_parameter),dalle_prompt_created,new_heading
