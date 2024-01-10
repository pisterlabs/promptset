import openai

def get_completion_from_messages(messages, 
                                 model="gpt-3.5-turbo", 
                                 temperature=0.5, 
                                 max_tokens=300):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, 
        max_tokens=max_tokens, 
    )
    return response.choices[0].message["content"]