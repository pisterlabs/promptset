# This is a financial expert chatbot that specializes in real estate investment and negotiation
|
|
|
# The code starts by importing the necessary libraries.
import openai
import gradio
# Gradio is the fastest way to demo your machine learning model with a friendly web interface
#  so anyone can use it, anywhere!


# It defines an openai API key that will be used to authenticate with the OpenAI platform.
openai.api_key = " Replace_with_Your_API_KEY "


# A list of messages is created and assigned to each role in the chatbot conversation.
messages = [{"role": "system",
             "content": "You are a financial expert that specializes in real estate investment and negotiation"}]

#  The first message is for when a user logs into the chatbot and starts talking to it.
# Next, a function called CustomChatGPT is defined which takes in text as input and returns text 
# as output.
# This function appends one of two things: either "user" or "assistant".
# In this case, we are using "user" because we want our bot to respond with what someone 
# says when they talk about their financial expertise or real estate investment negotiation skills.
# In this case, we are using "user" because we want our bot to respond with what someone says when 
# they talk about their financial expertise or real estate investment negotiation skills.
# The next step is creating an interface object called demo that has been given a custom function 
# named CustomChatGPT .
# Finally, this interface object needs to be launched so that it can start running on its own without
#  any further instructions from us!
# The code attempts to be executed by the user, who will input text into the "text" input of the
#  custom chat interface.
# The code will then create a response with a message containing the inputted text.


def CustomChatGPT(Query):
    messages.append({"role": "user", "content": Query})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": ChatGPT_reply})
    return ChatGPT_reply


demo = gradio.Interface(fn=CustomChatGPT, inputs="text",
                        outputs="text", title="Real Estate Master")

demo.launch(share=True)
