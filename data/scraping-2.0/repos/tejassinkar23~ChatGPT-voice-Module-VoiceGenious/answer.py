import openai
from apikey import api_data
openai.api_key = api_data
completion = openai.Completion()

def generate_response(question):
    prompt = f'Tejas: {question}\n Jarvis: '
    response = completion.create(prompt=prompt, engine="text-davinci-002", stop=['\Tejas'])
    answer = response.choices[0].text.strip()
    return answer
    
ans = generate_response("What is openAI?")
print(ans)                                     
