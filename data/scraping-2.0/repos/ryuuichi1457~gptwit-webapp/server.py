from flask import Flask 
""" render_template,request """
import openai
import datetime

app = Flask(__name__, static_folder='./static', static_url_path='')

def url_encode(string):
    string = string.replace("#", "%0a%23")
    string = string.replace(" ", "%20")
    string = string.replace("\n", "%0a")
    string = string.replace("「", "")
    string = string.replace("」", "")
    string = string.replace('"', "")
    return string

""" def make_Tweet(API_KEY,contents):
    try:
        current_datetime = datetime.datetime.now()
        time = current_datetime.strftime("%m-%d %H:%M")
        instruction=f"命令\n\n以下の条件、内容に沿ったツイート文をひとつ生成してください。\n\n条件\n\n生成する文章は140字以内にしてください。\nハッシュタグ、絵文字なども随時使ってください。\n現在時刻にあった内容を生成してください\n現在時刻は  {time}  です。\nリンクに使えない文字は使わないでください(#、スペース、改行、絵文字を除く)\n\nテーマ\n"
        ask = f"{instruction}{contents}"
        openai.api_key = API_KEY
        response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                    {'role': 'user', 'content': ask}],
                    max_tokens=150,
                    temperature=1.0,
        )
        url ="https://twitter.com/intent/tweet?text=" + url_encode(response['choices'][0]['message']['content'])
        atag=f"<a href=\"{url}\">{url}</a>"
        return(atag)
    except:
        return ("<link rel=\"stylesheet\" href=\"{{ url_for('static', filename='stylesheet.css') }}\">\n"
        "<p class=\"errormesseage\">エラーが発生しました。<br>APIキーが違ったり、連続した生成が原因であることが多いです。</p>")
 """
@app.route('/')
def index():
  """ return '<h1>Hello World</h1>' """
  return app.send_static_file('main.html')

@app.route('/test')
def hello():
   return '<h1>test yattane!!</h1>'

""" @app.route('/response/',methods=['POST'])
def response():
  API_KEY = request.form.get('API_KEY')
  content = request.form.get('content')
  return make_Tweet(API_KEY,content) """

if __name__ == "__main__":
    app.run()