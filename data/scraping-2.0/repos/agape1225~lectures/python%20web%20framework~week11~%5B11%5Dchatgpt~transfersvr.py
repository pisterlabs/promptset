import os
import openai
from flask import Flask, request

openai.api_key = "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" 

app = Flask(__name__)

@app.route('/')
def index():
    html= """
        <form action=/gpt > 
            <textarea name=txt rows=5 cols=50 >여우가 게으른 개를 뛰어 넘었다</textarea> <br/>
            <select name=mode>
                <option>영어</option>
                <option>낚시</option>
                <option>flask</option>
                <option>random</option>
            </select>

            <input type=submit value=전송 />
        </form>
    """    
    return html

@app.route('/gpt')
def gpt():    
    txt = request.args.get("txt", "안녕")    
    mode = request.args.get("mode", "영어")    
    change = {
    "낚시":"다음 문장을 낚시성 스타일로 바꿔주세요 ",
    "영어" : "다음 문장을 영어로 번역해 주세요 ",
    "flask" : "다음 문장을 출력하는 플라스크 코드를 출력해줘",
    "random" : "다음 문장을 문자 한글자 별로 순서를 뒤죽박죽 섞어줘"
    }


    prompt = change[mode] + "\n" + txt

    messages = []

    messages.append({"role": "user", "content": prompt})
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  messages=messages)

    res= completion.choices[0].message['content']

    html = f"""
     원문 : <p>{txt}</p></br>
     변환({mode}) :   {res}
    """   
    
    return html

    
if __name__ == '__main__':
	app.run(debug=True)