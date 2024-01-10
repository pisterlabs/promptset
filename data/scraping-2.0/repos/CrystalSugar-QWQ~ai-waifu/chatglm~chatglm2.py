import openai
import requests, json
import chatglm

history = []
conversation_file = "./conversation.json"    # 历史文件地址


# 调用chatglm接口，获取返回内容
def chatglm2_requset(prompt):
    global history
    send_data = {
        "prompt": prompt, 
        "history": history,
        "max_length": 2048,
        "top_p": 0.7,
        "temperature": 0.95
    }

    try:
        response = requests.post(url="http://localhost:8000", json=send_data)
        response.raise_for_status()  # 检查响应的状态码

        result = response.content
        ret = json.loads(result)

        resp_content = ret['response']

        history.append(ret['history'][0])

        return resp_content
    except Exception:
        return None


def chatglm2_answer(text, message_data):
    global history
    data = ""
    message = ""
    history.append({'role': 'user', 'content': text})
    prompt = chatglm.getPrompt(history)
    openai.api_base = "http://localhost:8000/v1"
    openai.api_key = "none"
    for chunk in openai.ChatCompletion.create(
        model="chatglm2-6b",
        messages=prompt,
        max_tokens=4096,
        temperature=1,
        top_p=0.9, 
        stream=True
    ):
        if hasattr(chunk.choices[0].delta, "content"):
            # print(chunk.choices[0].delta.content, end="", flush=True)
            data += chunk.choices[0].delta.content
            if chunk.choices[0].delta.content in r"。！？）.!?)~":
                # print(data)
                message_data.put(data)
                message += data
                data = ""
    history.append({'role': 'assistant', 'content': message})
    with open(conversation_file, "w", encoding="utf-8") as f:
        # Write the message data to the file in JSON format
        json.dump(history, f, indent=4)
    return message