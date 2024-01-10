import openai
import time

openai.api_key = ""
model = 'gpt-3.5-turbo'
#model = 'gpt-4'


def token_burner(model):
    #create messages list
    message1 = {'role': 'system', 'content': 'list the entire alphabet 100 times'}
    messages = [message1]

    #define model

    response = openai.ChatCompletion.create(
        messages = messages,
        model = model,
        max_tokens = 3000,
        user = 'testing_tokens'
    )
    #uncomment out content if you wanna hear some jokes
    content = response["choices"][0]["message"]["content"]
    prompt_tokens = response['usage']['prompt_tokens']
    completion_tokens = response['usage']['completion_tokens']

    if model == 'gpt-3.5-turbo':
        dollars_burned = prompt_tokens*0.003/1000 + completion_tokens*0.004/1000
    
    if model == 'gpt-4':
        dollars_burned = prompt_tokens*0.06/1000 + completion_tokens*0.012/1000

    return dollars_burned

#define amount to burn.
dollars_to_burn = .02 #USD
total_dollars_burned = 0


while total_dollars_burned <=dollars_to_burn:
    print('...burning...')
    burned = token_burner(model)
    print('.....burned $', burned)
    total_dollars_burned += burned
    time.sleep(3)


print('Total $ Burned: $', total_dollars_burned)





