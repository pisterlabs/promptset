import openai

openai.api_key = "YOUR_API_KEY_HERE"

def get_severity_chatgpt(user_input):
    question_to_ask = f"Strictly classify the complaint in quotes into the following severity groups: Very Severe, Moderately Severe, and Not Severe. Ensure that only the category is returned. No other additional text: '{user_input}'."
    response = openai.ChatCompletion.create(\
        model = "gpt-3.5-turbo",
        messages = [{"role": "system", "content": "You are a chatbot"},
				{"role": "user", "content": question_to_ask},
        ])

    result = ''
    for answer in response.choices:
        result += answer.message.content
    print(result)
    return result

user_input = "I have diabetes and I am running out of breath. I can't breathe properly and I have fainted 2 times today."
get_severity_chatgpt(user_input)