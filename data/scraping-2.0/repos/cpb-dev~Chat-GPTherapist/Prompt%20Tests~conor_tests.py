import openai, json

openai.api_key = 'Your API Key'

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.75,
    )
    return response.choices[0].message["content"]

def processInfo():
    inp = input("Type in analysis:\n")

    prompt = f"""
        Your role is to summarize the text delimited by triple backticks.
        Create a JSON file that contains the summary of the text, possible mental
        conditions it may relate to with a probability index.

        User input: ```{inp}```
    """
    response = get_completion(prompt)
    
    print(response)
    inp = input("ok?")

def readJSON():
    json_file = open("Prompt Tests/test.JSON")
    read_file = json.load(json_file)
    
    prompt =  f"""
        Your role is to look at the json file delimitec by triple backticks.
        Through your understanding, write a sentence explaining what a person might 
        need to look into to help the person being described.

        JSON file: '''{read_file}'''
    """

    response = get_completion(prompt)
    
    print(response)
    inp = input("ok?")

readJSON()
#processInfo()