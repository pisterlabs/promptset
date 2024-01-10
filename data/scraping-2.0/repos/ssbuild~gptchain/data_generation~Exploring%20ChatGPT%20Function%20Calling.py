#!/usr/bin/env python
# coding: utf-8

# # Function Calling with OpenAI's GPT Models: An Interactive Tutorial
# 
# In this notebook, we'll dive deep into a powerful feature offered by the latest versions of OpenAI's GPT models (like gpt-3.5-turbo-0613 and gpt-4-0613): function calling.
# 
# Let's imagine you're talking to a ChatGPT model and you want to have it use a tool. Traditionally you'd have to do some clever prompting to have it return the format you'd like.
# 
# Now you can tell it about certain actions, or **"functions"**, it can take
# 
# This doesn't mean the assistant actually performs these actions. Rather, it's aware of them and can instruct you on how to perform these actions based on the conversation at hand.
# 
# For example, you can tell the assistant about a function that fetches weather data, and when asked "What's the weather like in Boston?", the assistant can reply with instructions on how to call this weather-fetching function with 'Boston' as the input.
# 
# **Function calling** enables us to leverage the model's natural language understanding to effectively turn human language into structured data or specific function calls in our code.
# 
# This capability is useful in numerous scenarios, from creating chatbots that can interact with other APIs, to automating tasks and extracting structured information from natural language inputs. See more information about [function calling](https://platform.openai.com/docs/guides/gpt/function-calling)
# 
# Let's explore and start by importing our packages

# In[1]:


# !pip install langchain --upgrade
# Version: 0.0.199 Make sure you're on the latest version

import langchain
import openai
import json

# Environment Variables
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY', 'YourAPIKeyIfNotSet')


# ## OpenAI Vanilla Example
# 
# Let's run through OpenAI's vanilla example of calling a weather API.
# 
# First let's define our functions. This is the meat and potatoes of the new update
# 
# Functions are specified with the following fields:
# 
# * **Name:** The name of the function.
# * **Description:** A description of what the function does. The model will use this to decide when to call the function.
# * **Parameters:** The parameters object contains all of the input fields the function requires. These inputs can be of the following types: String, Number, Boolean, Object, Null, AnyOf. Refer to the API reference docs for details.
# * **Required:** Which of the parameters are required to make a query. The rest will be treated as optional.

# In[2]:


function_descriptions = [
            {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "description": "The temperature unit to use. Infer this from the users location.",
                            "enum": ["celsius", "fahrenheit"]
                        },
                    },
                    "required": ["location", "unit"],
                },
            }
        ]


# Then let's call the OpenAI API with this as a new parameter. Note: Make sure to use a model that can accept the function call. Here we are using `gpt-3.5-turbo-0613`.
# 
# Let's first set a query that came from the user

# In[3]:


user_query = "What's the weather like in San Francisco?"


# Then let's set up our API call to OpenAI. Note: `function_call="auto"` will allow the model to choose whether or not it responds with a function. You can set this to `none` if you *don't* want a function response

# In[4]:


response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        
        # This is the chat message from the user
        messages=[{"role": "user", "content": user_query}],
    
        
        functions=function_descriptions,
        function_call="auto",
    )


# Great, let's take a look at the response

# In[10]:


ai_response_message = response["choices"][0]["message"]
print(ai_response_message)


# Awesome, now we have our response w/ specific arguments called out.
# 
# Let's clean up our response a bit better

# In[11]:


user_location = eval(ai_response_message['function_call']['arguments']).get("location")
user_unit = eval(ai_response_message['function_call']['arguments']).get("unit")


# Then let's make a function that will serve as an interface to a dummy api call

# In[12]:


def get_current_weather(location, unit):
    
    """Get the current weather in a given location"""
    
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)


# In[13]:


function_response = get_current_weather(
    location=user_location,
    unit=user_unit,
)


# In[14]:


function_response


# Now that we have our reponse from our service, we can pass this information back to our model for a natural language response

# In[15]:


second_response = openai.ChatCompletion.create(
    model="gpt-4-0613",
    messages=[
        {"role": "user", "content": user_query},
        ai_response_message,
        {
            "role": "function",
            "name": "get_current_weather",
            "content": function_response,
        },
    ],
)


# In[17]:


print (second_response['choices'][0]['message']['content'])


# ## LangChain Support For Functions

# In[18]:


from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, ChatMessage
from langchain.tools import format_tool_to_openai_function, YouTubeSearchTool, MoveFileTool


# Let's load up our models

# In[19]:


llm = ChatOpenAI(model="gpt-4-0613")


# Let's load our tools and then transform them into OpenAI's function framework

# In[20]:


tools = [MoveFileTool()]
functions = [format_tool_to_openai_function(t) for t in tools]


# Let's take a look at what this tool was transformed as

# In[21]:


functions


# In[26]:


message = llm.predict_messages([HumanMessage(content='move file foo to bar')], functions=functions)


# In[28]:


message.additional_kwargs['function_call']


# ### Ad Hoc Example Financial Forecast Edit

# I'm going to make a new function description that talks about updating a financial model. It'll take 3 params, year to update, category to update, and amount to update.

# In[29]:


function_descriptions = [
            {
                "name": "edit_financial_forecast",
                "description": "Make an edit to a users financial forecast model",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "year": {
                            "type": "integer",
                            "description": "The year the user would like to make an edit to their forecast for",
                        },
                        "category": {
                            "type": "string",
                            "description": "The category of the edit a user would like to edit"
                        },
                        "amount": {
                            "type": "integer",
                            "description": "The amount of units the user would like to change"
                        },
                    },
                    "required": ["year", "category", "amount"],
                },
            },
            {
                "name": "print_financial_forecast",
                "description": "Send the financial forecast to the printer",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "printer_name": {
                            "type": "string",
                            "description": "the name of the printer that the forecast should be sent to",
                            "enum": ["home_printer", "office_printer"]
                        }
                    },
                    "required": ["printer_name"],
                },
            }
        ]


# One of the cool parts about OpenAI's new function calls is that the LLM will decide if it should return a normal response to a user, or call the function again. Let's test this out with two different requests in the same query from the user

# In[30]:


user_request = """
Please do three things add 40 units to 2023 headcount
and subtract 23 units from 2022 opex
then print out the forecast at my home
"""


# We are going to keep track of the message history ourselves. As more support for function conversations comes in we won't need to do this.
# 
# First we'll send the message from the user to the LLM along with our function calls

# In[31]:


first_response = llm.predict_messages([HumanMessage(content=user_request)],
                                      functions=function_descriptions)
first_response


# As you can see we get an AIMessage back with no content. However there are `additoinal_kwargs` with the information that we need. Let's pull these out to have a better look at them

# In[32]:


first_response.additional_kwargs


# In[33]:


function_name = first_response.additional_kwargs["function_call"]["name"]
function_name


# Then print the arguments it gives back to us

# In[34]:


print (f"""
Year: {eval(first_response.additional_kwargs['function_call']['arguments']).get('year')}
Category: {eval(first_response.additional_kwargs['function_call']['arguments']).get('category')}
Amount: {eval(first_response.additional_kwargs['function_call']['arguments']).get('amount')}
""")


# But we aren't done! There was a second request in the user query so let's pass it back into the model

# In[36]:


second_response = llm.predict_messages([HumanMessage(content=user_request),
                                        AIMessage(content=str(first_response.additional_kwargs)),
                                        ChatMessage(role='function',
                                                    additional_kwargs = {'name': function_name},
                                                    content = "Just updated the financial forecast for year 2023, category headcount amd amount 40"
                                                   )
                                       ],
                                       functions=function_descriptions)


# Let's see the response from this one

# In[37]:


second_response.additional_kwargs


# In[38]:


function_name = second_response.additional_kwargs['function_call']['name']
function_name


# Cool! It saw that the first response was done and then it went back to our function for us. Let's see what it says if we do it a third time

# In[39]:


third_response = llm.predict_messages([HumanMessage(content=user_request),
                                       AIMessage(content=str(first_response.additional_kwargs)),
                                       AIMessage(content=str(second_response.additional_kwargs)),
                                       ChatMessage(role='function',
                                                    additional_kwargs = {'name': function_name},
                                                    content = """
                                                        Just made the following updates: 2022, opex -23 and
                                                        Year: 2023
                                                        Category: headcount
                                                        Amount: 40
                                                    """
                                                   )
                                       ],
                                       functions=function_descriptions)


# In[40]:


third_response.additional_kwargs


# In[42]:


function_name = third_response.additional_kwargs['function_call']['name']
function_name


# Nice! So it knew it was done with the financial forecasts (because we told it so) and then it sent our forecast to our home printer. Let's close it out

# In[43]:


forth_response = llm.predict_messages([HumanMessage(content=user_request),
                                       AIMessage(content=str(first_response.additional_kwargs)),
                                       AIMessage(content=str(second_response.additional_kwargs)),
                                       AIMessage(content=str(third_response.additional_kwargs)),
                                       ChatMessage(role='function',
                                                    additional_kwargs = {'name': function_name},
                                                    content = """
                                                        just printed the document at home
                                                    """
                                                   )
                                       ],
                                       functions=function_descriptions)


# In[44]:


forth_response.content


# In[ ]:




