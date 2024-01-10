from flask import Flask, request
import openai
import hashlib
import time
import xml.etree.ElementTree as ET

app = Flask(__name__)
app.config['your_api_secret'] = 'your_token'  # 将your_token替换为你的Token字符串

openai.api_key = "your_api_key"  # 将your_api_key替换为你的OpenAI API Key


# 定义接收微信服务器消息的路由
@app.route('/wechat', methods=['GET', 'POST'])
def wechat():
    # GET请求是用来验证服务器的
    if request.method == 'GET':
        # 获取请求参数
        signature = request.args.get('signature', '')
        timestamp = request.args.get('timestamp', '')
        nonce = request.args.get('nonce', '')
        echostr = request.args.get('echostr', '')
        # 替换为自己的Token字符串
        token = app.config['your_api_secret']
        data = [token, timestamp, nonce]
        data.sort()
        # 将数据拼接成字符串并进行SHA1加密
        s = ''.join(data).encode('utf-8')
        if signature == hashlib.sha1(s).hexdigest():
            # 如果验证通过，则返回echostr给服务器
            return echostr
        else:
            # 如果验证不通过，则返回错误信息
            return 'Token verification failed'
    # POST请求是用来接收用户发送的消息
    elif request.method == 'POST':
        # 解析XML格式的数据
        xml_str = request.data.decode('utf-8')
        xml = ET.fromstring(xml_str)
        msg_type = xml.find('MsgType').text
        from_user = xml.find('FromUserName').text
        to_user = xml.find('ToUserName').text
        if msg_type == 'text':
            content = xml.find('Content').text
            # 使用OpenAI API进行自然语言处理
            response = openai.Completion.create(
                engine="davinci",
                prompt=content,
                max_tokens=50,
                n=1,
                stop=None,
                temperature=0.5,
            )
            # 获取AI返回的回复内容
            reply_content = response.choices[0].text.strip()
            # 构建XML格式的回复消息
            reply_xml = f'''
            <xml>
                <ToUserName><![CDATA[{from_user}]]></ToUserName>
                <FromUserName><![CDATA[{to_user}]]></FromUserName>
                <CreateTime>{int(time.time())}</CreateTime>
                <MsgType><![CDATA[text]]></MsgType>
                <Content><![CDATA[{reply_content}]]></Content>
            </xml>
            '''
            # 将回复消息返回给微信服务器
            return reply_xml
        else:
            return 'success'


# 启动Flask应用
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
