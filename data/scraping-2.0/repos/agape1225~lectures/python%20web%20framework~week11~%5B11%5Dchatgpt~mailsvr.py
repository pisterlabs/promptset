import os
import openai
from flask import Flask, request

openai.api_key = "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" 

app = Flask(__name__)

@app.route('/')
def index():
    html= """
        <form action=/gpt > 
            주제 <input type=text name=subject size=80> <br/>
            to <input type=text name=to > <br/>
            <select name=style>
                <option value="안부를 물어보는 통상적인 문장으로 작성">안부</option>
                <option value="서운한 감정이 드러나도록 작성">서운한감정</option>
                <option value="오해를 풀고 사과하는 마음을 담아서 작성">사과</option>                
            </select>
            <input type=submit value=메일작성 />
        </form>
    """    
    return html

@app.route('/gpt')
def gpt():    
    subject = request.args.get("subject", "안녕")    
    to  = request.args.get("to", "친구")    
    style = request.args.get("style", "안부")       
    
    prompt = f"""
    당신은 글을 매우 잘쓰는 작가입니다. 
    아래와 같은 내용으로 {style} 메일을 작성해주세요.
    내용 :  {subject}
    대상 :  {to}
    """
    messages = []

    messages.append({"role": "user", "content": prompt})
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  messages=messages)    

    res= completion.choices[0].message['content']
    print(res)   

    html = f"""
    {subject} <br/> 
    to : {to}<hr/>
    {res}    
    """ 
    return html

    
if __name__ == '__main__':
	app.run(debug=True)