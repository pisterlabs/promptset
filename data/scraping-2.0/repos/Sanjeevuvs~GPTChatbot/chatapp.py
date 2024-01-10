import openai
import config

openai.api_key= config.DevelopmentConfig.OPENAI_KEY

def generateChatResponse(prompt):
    messages = []
    messages.append({"role": "system", "content": "Your name is Sanjeev.you are a helpful assistant."})

    question = {}
    question['role'] = 'user'
    question['content'] = prompt
    messages.append(question)
    engine = "text-davinci-003"

    # response=openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=messages)
    response = openai.Completion.create(engine=engine, prompt=prompt, max_tokens=1024, n=1, stop=None,
                                        temperature=0.7, )

    try:
        #answer=response['choices'][0]['message']['content'].replace('\n','<br>')
        resanswer = response
        answer = resanswer.choices[0].text.strip().replace('\n', '<br>')
    except:
        answer = "Oops you beat the AI,try a different question, if the problem presists,come back later"

    return answer
