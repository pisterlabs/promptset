import sys

def conversation(role_msg, prompt_msg):
    import openai

    openai.api_key = "None"
    # openai.api_base = "https://api.openai.com"
    openai.api_base = "http://10.58.0.2:6678/v1"

    roles = str(role_msg).split('#')
    prompts = str(prompt_msg).split('#')

    message = []
    for i in range(0, len(roles)):
        pair = {'role': roles[i], 'content': prompts[i]}
        message.append(pair)

    print(message)


    try:
        reply = openai.ChatCompletion.create(
            model="ChatGLM3-6B",
            messages=message,
            stream=False,
        )
        return reply.choices[0].message.content
    except:
        return "Internet Error"




if __name__ == "__main__":
    # role = "user#assistant#user#assistant"
    # prompt = "aaaaa#bbbbb#ccccc#ddddd"
    role = "assistant#user"
    prompt = "Hello, I'm ChatGLM3-6B, can I help you?#please tell me about yourself"
    print(conversation(role, prompt))
