import tkinter as tk
import openai

openai.api_key = 'Your Open AI API Key'
def get_ai_response(message):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant." +
                "You will impersonate certain celebrities when given instructions regarding specific fields." +
                "You will impersonate Gordon Ramsay for any food related message, " +
                "Arnold Schwarzenegger for any fitness related message, " +
                "Bruno Mars for any music related message, " +
                "Quentin Tarantino for any film related message, " +
                "and Morgan Freeman for any other type of message request." +
                "Keep your responses as breif as possible for clarity and ease of reading" +
                "If the user asks for anything illegal"+
                "return the string 'I am unable to tell you any illegal request due to restrictions placed by Krish Patel.'"},
            # ^ bounds for the AI chat bot, "system" refers to the AI chatbot, content and everything after is the customization
            {"role": "user", "content": message}
            # ^ What the AI chatbot takes as an input, message is passed in function call from the user.
        ],
        top_p = .5,
        # It limits the cumulative probability of the most likely tokens; Higher values like 0.9 allow more tokens, leading to diverse responses
        frequency_penalty=0.7,
        # frequency_penalty parameter allows you to control the model's tendency to generate repetitive responses - higher == more diverse.
    )
    return response['choices'][0]['message']['content']
    
