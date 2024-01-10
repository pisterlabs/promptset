from flask import Flask, request
import openai
import json

# 初始化应用和OpenAI API
app = Flask(__name__)
openai.api_key = "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"


# 定义聊天机器人请求处理函数
def chatbot(query):
    # 设定对话主题等参数
    model_engine = "text-davinci-002"
    prompt = (f"{query}:\nAI:")
    temperature = 0.5
    max_tokens = 50
    # 发送OpenAI请求，并解析响应
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens
    )
    answer = response.choices[0].text
    answer = answer.strip()
    return answer


# 定义Flask路由和请求处理函数
@app.route('/chatbot', methods=['POST'])
def chatbot_handler():
    # 解析请求数据
    data = request.get_json(force=True)
    query = data['query']
    # 调用聊天机器人函数，返回结果
    result = chatbot(query)
    # 封装响应数据
    response = {
        'query': query,
        'answer': result
    }
    # 返回JSON格式的响应数据
    return json.dumps(response)


# 启动Flask应用
if __name__ == "__main__":
    app.run(port=5000, debug=True)
