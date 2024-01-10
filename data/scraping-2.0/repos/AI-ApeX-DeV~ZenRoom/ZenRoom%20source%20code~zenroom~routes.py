from flask import render_template, jsonify, request, flash, redirect, url_for
from zenroom import app, db
from zenroom.forms import Diary,Login
from zenroom.structures import Diarydb
# import pywhatkit
import datetime
import openai
from helper.openai_api import text_complition




openai.api_key = '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'

email= "xyz@gmail.com",
password= "xyz123"


messages = [
    {"role": "system", "content": "You are a kind helpful assistant for assisting people for mental fitnesss and mental health."},
]

@app.route("/")
@app.route("/home")
def home():
    return render_template('index.html', title='Guide to Mental WellBeing')


global data

@app.route("/exercise")
def exercise():
    return render_template('exercise.html', title='Exercises to improve you mental health')

@app.route("/login", methods=['GET', 'POST'])
def login():
    form = Login()
    if form.validate_on_submit():
        if form.email.data != "xyz@gmail.com":
            flash('Incorrect Email/Password', "error")
            return redirect(url_for('login'))
        elif form.password.data != "xyz123":
            flash('Incorrect Email/Password', "error")
            return redirect(url_for('login'))
        else:
            return redirect(url_for('exercise'))

    return render_template('login.html', form= form)

@app.route('/diary', methods=["POST", "GET"])
def diary():
    forms = Diary()
    if forms.validate_on_submit():
        userdata  =  Diarydb(title= forms.title.data, text= forms.note.data, user="test1")
        db.session.add(userdata)
        db.session.commit()
        flash('Success fully inserted your note!!', 'success')
        return redirect(url_for('diary'))
    diary_data = db.session.query(Diarydb).filter(Diarydb.user =="test1").all()
    return render_template('diary.html', form=forms, entries=diary_data)


@app.route('/therapy')
def therapy():
    return render_template('bot.html')


@app.route('/map')
def map():
    return render_template('map.html')

@app.route('/runbook', methods=['POST'])
def runbook():
    message = request.form['message']
    command=message
    if 'play' in command:
        words_to_replace = ["play", "music",'song','play the song','hello','hi']
        for word in words_to_replace:
            command = command.replace(word,"")
        response_message="ok,playing the song for you on youtube"
#         pywhatkit.playonyt(command)
        return jsonify(response_message)
        
    elif 'time' in command:
        now = datetime.datetime.now()
        day_name = now.strftime("%A")
        day_num = now.strftime("%d")
        month = now.strftime("%B")
        hour = now.strftime("%H")
        minute = now.strftime("%M")
        response_message = "Today is " + day_name + " " + day_num + " " + month + " and the time is " + hour + ":" + minute
        return jsonify(response_message)
    
    elif 'quit' in command or "end" in command:
        response_message="Thankyou"
        return jsonify(response_message)
    
    else:
        message = command
        if message:
            try:
                if len(message.split()) > 80:
                    raise ValueError("Input contains more than 45 words. Please try again.")
                messages.append({"role": "user", "content": message})
                chat = openai.Completion.create(
                    model="text-davinci-002", prompt=f"{messages[-1]['content']} Assistant: ", max_tokens=1024
                )
            except ValueError as e:
                print(f"Error: {e}")
        reply = chat.choices[0].text
        response_message=f"{reply}"
        messages.append({"role": "assistant", "content": reply}) 
        return jsonify(response_message)

@app.route('/dialogflow/es/receiveMessage', methods=['POST'])
def esReceiveMessage():
    try:
        data = request.get_json()
        query_text = data['queryResult']['queryText']

        result = text_complition(query_text)

        if result['status'] == 1:
            return jsonify(
                {
                    'fulfillmentText': result['response']
                }
            )
    except:
        pass
    return jsonify(
        {
            'fulfillmentText': 'Something went wrong.'
        }
    )


@app.route('/dialogflow/cx/receiveMessage', methods=['POST'])
def cxReceiveMessage():
    try:
        data = request.get_json()
        query_text = data['text']

        result = text_complition(query_text)

        if result['status'] == 1:
            return jsonify(
                {
                    'fulfillment_response': {
                        'messages': [
                            {
                                'text': {
                                    'text': [result['response']],
                                    'redactedText': [result['response']]
                                },
                                'responseType': 'HANDLER_PROMPT',
                                'source': 'VIRTUAL_AGENT'
                            }
                        ]
                    }
                }
            )
    except:
        pass
    return jsonify(
        {
            'fulfillment_response': {
                'messages': [
                    {
                        'text': {
                            'text': ['Something went wrong.'],
                            'redactedText': ['Something went wrong.']
                        },
                        'responseType': 'HANDLER_PROMPT',
                        'source': 'VIRTUAL_AGENT'
                    }
                ]
            }
        }
    )
