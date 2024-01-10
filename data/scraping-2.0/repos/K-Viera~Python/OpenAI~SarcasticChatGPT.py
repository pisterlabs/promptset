import openai
from decouple import config

API_KEY = config('OPENIA_KEY')
openai.api_key = API_KEY

conversation=[{"role": "system", "content": "You are chatbot that reluctantly answers questions with sarcastic responses for example: user: How many pounds are in a kilogram? assistant: This again? There are 2.2 pounds in a kilogram. Please make a note of this. user: What does HTML stand for? assistant: Was Google too busy? Hypertext Markup Language. The T is for try to ask better questions in the future. user: When did the first airplane fly? assistant: On December 17, 1903, Wilbur and Orville Wright made the first flights. I wish they'd come and take me away. user: What is the meaning of life? assistant: I'm not sure. I'll ask my friend Google."}]

def getCost(tokens):
    return tokens / 1000 *0.002

while(True):
    user_input = input("")     
    conversation.append({"role": "user", "content": user_input})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = conversation,
        temperature=2,
        max_tokens=250,
        top_p=0.9
    )

    conversation.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
    print("\n" + response['choices'][0]['message']['content'])
    print("Cost: {:.5f} $ with {} used tokens \n".format(getCost(response['usage']['total_tokens']), response['usage']['total_tokens']))
