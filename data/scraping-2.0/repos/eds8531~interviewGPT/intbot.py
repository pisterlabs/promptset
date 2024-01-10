import openai
from chatbot2 import complete_prompt, get_completion_from_messages, collect_messages
from flask import Flask, render_template, request

app = Flask(__name__)

openai.api_key = "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" 

@app.route('/', methods=['GET', 'POST'])
def interview():
    if request.method == 'POST':
        job = request.form['job']
        company = request.form['company']
        requirements = request.form['requirements']
        context = complete_prompt(job, company, requirements)
        return render_template('chat.html', context=context, message='')
    else:
        return render_template('index.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        job = request.form['job']
        company = request.form['company']
        requirements = request.form['requirements']
        context = complete_prompt(job, company, requirements)
        try:
            message = request.form['message']
            response, context = collect_messages(context, message)
            return render_template('chat.html', chatbot_response = response, context=context, message = message)
    
        except:
            response, context = collect_messages(context)
            return render_template('chat.html', context=context, chatbot_response = response)
    else:
        return render_template('index.html')
    
@app.route('/get')
def get_bot_response():
    userText = request.args.get('msg')
    return str(bot.get_response(userText))
            
    

if __name__ == '__main__':
    app.run(debug=True)