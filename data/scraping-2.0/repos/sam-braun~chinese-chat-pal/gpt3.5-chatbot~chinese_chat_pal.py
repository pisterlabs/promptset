from flask import Flask, render_template, request, jsonify
import openai
from datetime import datetime
import jieba
from pypinyin import lazy_pinyin, Style
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

openai.api_key = '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'

messages = [{"role": "system", "content": "You are a helpful assistant."}]

def chinese_to_pinyin(chinese_text):
    logging.getLogger('jieba').setLevel(logging.ERROR)
    segments = jieba.lcut(chinese_text)
    return ' '.join([''.join(lazy_pinyin(segment, style=Style.TONE)) for segment in segments])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json['user_message']

    # User-selected options
    show_pinyin = request.json['show_pinyin']
    show_simplified = request.json['show_simplified']

    chinese_char_style = 'Simplified' if show_simplified else 'Traditional'
    instruction = "Please respond in " + chinese_char_style + " Mandarin."

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages + [{"role": "user", "content": user_input + f' ({instruction})'}]
    )

    # get Chinese answers
    answer_chinese = response.choices[0].message['content']
    messages.append({"role": "assistant", "content": answer_chinese})
    answer_pinyin = chinese_to_pinyin(answer_chinese)
    
    return jsonify({'chinese': answer_chinese, 'pinyin': answer_pinyin})

if __name__ == '__main__':
    app.run(debug=True)
