import openai

def gpt_informations(prompt):

    arquivo = open("key/key.txt", "r")
    ki = arquivo.readline()

    openai.api_key = f"{ki}"

    model_engine = "text-davinci-002"

    completion = openai.Completion.create(
        engine=model_engine, 
        prompt=format(prompt), 
        max_tokens=1024, 
        n=1,stop=None,
        temperature=0.5)


    return completion
