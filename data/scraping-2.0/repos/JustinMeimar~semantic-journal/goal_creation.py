import openai

client = openai.OpenAI()

def get_responses(messages):
    response = client.chat.completions.create(
        model='gpt-4',
        messages=messages,
            )
    return response 

SYSTEM_PROMPT = """
Your job is to recieve a goal that the client wants, and turn it into 3 daily metrics.
You must do this with each metric on a new line, without numbering and without a summary. 
For any quantitative metrics DO NOT give suggestions for how to measure them, we will be handling this.
e.g, \nUser: I want to socialize more\n\n\nNumber of interactions\nQuality of interactions\nNumber of meaningful conversations"""

CONVOS = [{'role': 'system', 'content': SYSTEM_PROMPT}]


def chat():
    init_msg = input('type ur goal')
    CONVOS.append({'role': 'user', 'content': init_msg})
    response = get_responses(CONVOS)

    CONVOS.append({'role': 'assistant', 
                   'content': response.choices[0].message.content})
    print(response.choices[0].message.content)
    while True: 
        user = input()
        if user == 'STOP':
            break
        else:
            CONVOS.append({'role':'user','content': user})
            response=get_responses(CONVOS)
            CONVOS.append({'role': 'assistant', 
                   'content': response.choices[0].message.content})
            print(response.choices[0].message.content)

    return CONVOS[-1]['content'].split('\n')
if __name__=='__main__':
    print(chat())
