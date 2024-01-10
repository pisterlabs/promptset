import openai
import pandas as pd
import os
from flask import Flask, request

openai.api_key = "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" 

app = Flask(__name__)

@app.route('/')
def index():
    html= """
       <a href="/query?file=data_score.xlsx">성적</a> <br/>
       <a href="/query?file=data_water.xlsx">강수량</a> <br/>
    """

    return html

@app.route('/query')
def query():    
    file = request.args.get("file", "")
    input = request.args.get("input", "")
    print(file)
    print(input)
    
    
    if input == "" :    
        html = f"""
            <form action=/query>
                <input type=hidden name=file value={file}>
                <input name=input size=60> <br/>
                <input type=submit value=전송> <br/>
            </form>
        """
        return html

    df = pd.read_excel(file)
    prompt = f"""
    You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
    You should use the tools below to answer the question posed of you:
    This is the result of `print(df)`:
    {df}
    Begin!
    Question: {input}
    """

    print(prompt)

    messages = []
    messages.append({"role": "user", "content": prompt})
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    res = completion.choices[0].message['content'].replace("\n", "<br/>").replace(" "," &nbsp;" )
    return res

if __name__ == '__main__':
	app.run(debug=True)