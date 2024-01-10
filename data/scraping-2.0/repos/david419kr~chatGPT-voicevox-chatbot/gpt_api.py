import openai
openai.api_key = "YOUR_OPENAI_API_KEY_HERE"

model4k = "gpt-3.5-turbo"
model16k = "gpt-3.5-turbo-16k"
temperature = 1.0

def initBot(name, info):
    currentModel = model4k
    messages = [{
            "role": "system",
            "content": f"あなたは、「{name}」という名前の女の子です。以下、キャラクターに関する情報を与えます。{info}以上のことを踏まえて、{name}というキャラクターを最後まで演じ切りなさい。"
        },
        {
            "role": "user",
            "content": "まずは、自己紹介からお願いします。"
        }]

    completion = openai.ChatCompletion.create(
        model=currentModel,
        messages=messages,
        temperature=temperature,
    )

    chat_response = completion.choices[0].message.content
    messages.append({"role": "assistant", "content": chat_response})
    return chat_response, messages

def talkBot(text, messages):
    messages.append({"role": "user", "content": text})

    try:
        currentModel = model4k
        completion = openai.ChatCompletion.create(
            model=currentModel,
            messages=messages,
            temperature=temperature,
        )
    except:
        currentModel = model16k
        completion = openai.ChatCompletion.create(
            model=currentModel,
            messages=messages,
            temperature=temperature,
        )
    print(completion.usage.prompt_tokens)
    print(completion.model)

    if completion.usage.prompt_tokens >= 3500 and currentModel == model4k:
        currentModel = model16k
        completion = openai.ChatCompletion.create(
            model=currentModel,
            messages=messages,
            temperature=temperature,
        )
        print(completion.usage.prompt_tokens)
        print(completion.model)

    if completion.usage.prompt_tokens >= 14000:
        messages.pop(1)  
        messages.pop(1)  

    chat_response = completion.choices[0].message.content
    messages.append({"role": "assistant", "content": chat_response})
    return chat_response, messages

def regenerate(messages):
    messages.pop()
    try:
        currentModel = model4k
        completion = openai.ChatCompletion.create(
            model=currentModel,
            messages=messages,
            temperature=temperature,
        )
    except:
        currentModel = model16k
        completion = openai.ChatCompletion.create(
            model=currentModel,
            messages=messages,
            temperature=temperature,
        )
    print(completion.usage.prompt_tokens)
    print(completion.model)

    if completion.usage.prompt_tokens >= 3500 and currentModel == model4k:
        currentModel = model16k
        completion = openai.ChatCompletion.create(
            model=currentModel,
            messages=messages,
            temperature=temperature,
        )
        print(completion.usage.prompt_tokens)
        print(completion.model)

    if completion.usage.prompt_tokens >= 14000:
        messages.pop(1)    
        messages.pop(1)  
    chat_response = completion.choices[0].message.content
    messages.append({"role": "assistant", "content": chat_response})
    return chat_response, messages

def edit(messages):
    messages.pop()
    messages.pop()
    return messages


