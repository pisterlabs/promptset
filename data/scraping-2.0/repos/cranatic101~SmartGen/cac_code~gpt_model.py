import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
def generate_recommendations(msgs):
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=msgs,
    temperature=1.21,
    max_tokens=600,
    top_p=1,
    frequency_penalty=0.3,
    presence_penalty=0.3
    )
    
    #removes empty values in the response list
    response_lst=convert_to_list(response["choices"][0]["message"]["content"])
    for i in response_lst:
        if i=='':
            response_lst.remove(i)

    return response["choices"][0]["finish_reason"],response_lst

def create_msgs(roles,contents):
    msgs = []
    for i in range(len(roles)):
        content = ""
        for line in contents[i]:
            content = content + line + "\n"
        content = content[:-2]
        msgs.append({"role":roles[i], "content": content})
    return msgs

def convert_to_list(string):
    lines = string.split('\n')
    for index, line in enumerate(lines):
        lines[index] = line.replace("\n", "")
    return lines