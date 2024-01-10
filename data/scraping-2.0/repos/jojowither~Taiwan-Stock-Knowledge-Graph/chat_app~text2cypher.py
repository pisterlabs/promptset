import os
import openai
from retry import retry
from dotenv import load_dotenv

from training import examples

load_dotenv()
openai.api_key  = os.getenv('OPENAI_KEY')
openai_model = os.getenv('OPENAI_MODEL')


system = f"""
你是一個能夠基於 Cypher queries 示例生成 Cypher queries的助手。示例 Cypher queries為：\n {examples} \n
除了 Cypher queries之外，請不要回答任何解釋或其他訊息，這點務必注意。
你不需要道歉，嚴格根據提供的 Cypher queries示例生成 Cypher 語句。
不提供任何無法從 Cypher queries示例推斷出來的 Cypher 語句。
當缺乏對話內容的上下文而無法推斷出 Cypher 語句時，請告知用戶缺少的上下文並說明原因。
只需要生成Cypher queries，不要其他文字訊息。
"""


@retry(tries=2, delay=5)
def generate_cypher(messages):
    messages = [
        {"role": "system", "content": system}
    ] + messages
    print(messages)
    # Make a request to OpenAI
    completions = openai.ChatCompletion.create(
        model=openai_model,
        messages=messages,
        temperature=0.0
    )
    response = completions.choices[0].message.content
    # Sometime the models bypasses system prompt and returns
    # data based on previous dialogue history
    if not "MATCH" in response and "{" in response:
        raise Exception(
            "GPT繞過了系統訊息，根據以前的對話記錄回覆。 " + response)
    # If the model apologized, remove the word
    if "抱歉" in response:
        response = " ".join(response[2:])
    # Sometime the model adds quotes around Cypher when it wants to explain stuff
    if "`" in response:
        response = response.split("```")[1].strip("`")
    print(response)
    return response


if __name__ == '__main__':
    generate_cypher([{'role': 'user', 'content': '查詢股票代碼2330基本資訊'}]) 
    generate_cypher([{'role': 'user', 'content': '列出廣達副總經理持股張數的基本統計'}]) 