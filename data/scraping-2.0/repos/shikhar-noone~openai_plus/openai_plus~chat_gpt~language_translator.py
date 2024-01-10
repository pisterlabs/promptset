import openai
openai.api_key = "sk-2R7ZTpdQwsgbiHrOvIsIT3BIbkFJ3Fd3hel91UzkDw07omI"

def translate_query(query, language):

    prompt = f"query = '{query}'./n Please translate the query into {language} and return statement only."
    completions = openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=2048, n=1,stop=None,temperature=0.5)
    print(completions)
    message = completions.choices[0].text
    #print(completions)
    return message



