from openai import OpenAI
import httpx


def read_properties(filename):
    properties = {}
    with open(filename, 'r') as f:
        for line in f:
            if "=" in line and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                properties[key.strip()] = value.strip()
    return properties


def talkToGpt(prompt):
    # 读取param.properties文件
    properties = read_properties('../param.properties')

    if properties.get('need_proxy') == 'Y':
        proxy_host = properties.get('proxyHost')
        proxy_port = properties.get('proxyPort')
        http_client = httpx.Client(proxies=f"http://{proxy_host}:{proxy_port}")
        client = OpenAI(http_client=http_client)
    else:
        client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )
    return completion.choices[0].message.content


def talkToGptStream(prompt):
    # 读取param.properties文件
    properties = read_properties('../param.properties')

    if properties.get('need_proxy') == 'Y':
        proxy_host = properties.get('proxyHost')
        proxy_port = properties.get('proxyPort')
        http_client = httpx.Client(proxies=f"http://{proxy_host}:{proxy_port}")
        client = OpenAI(http_client=http_client)
    else:
        client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        stream=True
    )
    final_output = ""
    for chunk in completion:
        if chunk is not None and chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            yield content
            final_output += content
    return final_output
