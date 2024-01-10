import openai


with open('openaiapikey.txt', 'r') as infile:
    open_ai_api_key = infile.read()
openai.api_key = open_ai_api_key


#ft = 'curie:ft-user-aev4t8pdy0qsyqxayyjuuqut-2021-08-16-14-23-55'
ft = 'curie:ft-user-aev4t8pdy0qsyqxayyjuuqut-2021-08-16-14-55-01'


def completion(prompt, model, temp=0.5, top_p=0.95, tokens=200, freq_pen=0.0, pres_pen=0.0, stop=['<<END>>']):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temp,
        max_tokens=tokens,
        top_p=top_p,
        frequency_penalty=freq_pen,
        presence_penalty=pres_pen,
        stop=stop)
    return response['choices'][0]['text'].strip().splitlines()

prompt = '''So guys I have this crush on this boy and he's so dumb but I love him. His name is Roger and he's the quarterback of the team.
QUESTIONS:'''

questions = completion(prompt, ft)
print(questions)