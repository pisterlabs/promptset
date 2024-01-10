import openai

with open('D:\Programs\RoundTable\python\indoct.txt', 'r') as f:
    indoctrination = f.readlines()[0]
    f.close

workingWithGPT = True
conversation = [{"role": "system", "content": indoctrination}]

openai.api_key_path = 'D:\KEYS\TextAdvKey.txt'

def promptGPT(userInput):
    global conversation
    conversation.append({"role": "user", "content": userInput})
    completion = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages = conversation
    )
    return completion.choices[0].message.content



while(workingWithGPT):
    userIn = input('>>>')
    if (userIn == 'stop'):
        workingWithGPT = False
    else:
        gptOut = promptGPT(userIn)
        conversation.append({"role": "system", "content": gptOut})
        print(gptOut)