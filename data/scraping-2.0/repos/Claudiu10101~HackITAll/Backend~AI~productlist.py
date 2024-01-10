import openai

def ai_answer(query,system_promt):
    openai.api_key = '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'
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_promt},
                {"role": "user", "content": query}
            ]
        )
        return response.choices[0].message['content']
    except Exception as e:
        return str(e)

def answer (sample):
    answer = input_string.split()
    return answer

