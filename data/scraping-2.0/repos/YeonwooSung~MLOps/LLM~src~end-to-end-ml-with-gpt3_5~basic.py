import openai

openai.api_key = "YOUR KEY HERE"

def get_api_result(prompt):
    request = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        messages=[{"role": "user", "content": prompt}]
    )
    
    result = request['choices'][0]['message']['content']

    print(result)
