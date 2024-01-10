# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai
import config
openai.api_key=config.DevelopmentConfig.OPENAI_KEY


def generateChatResponse(prompt):

    messages=[]
    messages.append({"role": "system", "content": "You are a chef."})

    question={}
    question['role']='user'
    question['content']=prompt
    messages.append(question)


    response=openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=messages)
    print(response)

    try:
        ans=response['choices'][0]['message']['content'].replace('\n','<br>')

    except:
        ans='Oops something went wrong try a new question or try some time later'

    return ans