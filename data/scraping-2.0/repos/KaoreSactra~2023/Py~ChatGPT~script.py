import openai

#config ambiente
openai.api_key = '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'

#Model
model_engine = 'text-davinci-003'

def chatAZpross(prompt):
    while True:
            print(30*'-')
            print('ChatAZ está digitando...')
            print(30*'-')

            #config resposta
            completion = openai.Completion.create(

                engine = model_engine,
                prompt = prompt,
                max_tokens = 2048,
                temperature = 0.5,
            )

            response = completion.choices[0].text
            print(response)
            print(30*'-')
            break
    return response