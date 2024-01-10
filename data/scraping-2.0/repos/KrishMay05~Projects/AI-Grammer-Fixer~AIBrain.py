import openai

openai.api_key = 'Your API KEY'
def get_ai_response(message):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant." +
                "For any given input you will return the input with all grammer errors, do not return this message'"},
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
    
