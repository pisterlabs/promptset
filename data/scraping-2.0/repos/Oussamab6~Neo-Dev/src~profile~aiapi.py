import openai
import confai
Api_key = confai.DevelopmentConfig.OPENAI_KEY

openai.api_key = Api_key


def generate_chatbot_response(prompt, initiprompt):
    print('entring')

    # initial_prompt = 'I want you to act as a chef. I will provide you with specific keywords, present arguments and specific guidelines or instructions. Your task is to write detailed responses to those keywords and provide me with specific examples of what should be said. You should only reply with the examples and nothing else. Do not write explanations. My first request is "I need help preparing dinner for 2 people."'
    message = [{"role": "system", "content": "system message"}, {
        "role": "user", "content": f"{initiprompt}\nUser: {prompt}"}]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=message,)
    try:

        answer = response['choices'][0]['message']['content'].replace(
            '\n', '<br>')

    except IndexError:
        answer = 'Please try a  diffrent question ! '
    return answer
