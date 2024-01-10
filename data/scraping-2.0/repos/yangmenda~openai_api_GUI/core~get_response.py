import openai


def get_res(s,model,key):
    
    openai.api_key = key    
    response = openai.Completion.create(

    model=model,

    prompt=s,

    temperature=0.5,

    max_tokens=2000)
    
    return response["choices"][0]["text"]

