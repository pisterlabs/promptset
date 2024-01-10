import os
import openai
import json

#loading key
with open('api-key.txt','r') as key:
    data = key.read().strip()
openai.api_key = data

# QUESTION_PROMPT = "Type your question here: "
# # Get user input
# question = input(QUESTION_PROMPT)


def asking_question(question):
    print("Asking ChatGPT this question: " + question)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Specify the chat model ("gpt-4.0" is recommended for the latest version)
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ],
            temperature= 0.7
            
        )
        answer = response.choices[0].message["content"]
        print('answer from chat gpt'+ str(answer))
        usage = response['usage']

        print("in this question we used: "+ str(usage['prompt_tokens']) + " prompt_tokens")
        print("in this question we used: "+ str(usage['completion_tokens']) + " completion_tokens")
        print("in this question we used: "+ str(usage['total_tokens']) + " total_tokens")
        
    except Exception as e:
        print("An error occurred:", str(e))
        answer = "An error occurred while processing your request."
       
    return answer
