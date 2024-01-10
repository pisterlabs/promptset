from flask import Flask, render_template, request, redirect, url_for,send_file, session, abort, flash, make_response, send_file
from PyDictionary import PyDictionary
from googletrans import Translator
import mysql.connector
import os
import pathlib
import requests
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
from pip._vendor import cachecontrol
from google.auth.transport.requests import Request
import google.auth.transport.requests
import base64
import PIL.Image as Image
import io
from io import BytesIO
from jinja2_base64_filters import jinja2_base64_filters
from jinja2 import Environment
import openai
import re

api_key = "AIzaSyBKSSLUiePvvR5XRXtTjcfSpQ61E2EDavw"

openai.api_key = ("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")

env = Environment(extensions=["jinja2_base64_filters.Base64Filters"])
env.filters['b64encode'] = base64.b64encode

os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

GOOGLE_CLIENT_ID = "656506681629-hmi68222d490h5km2fqe8sm783as3e27.apps.googleusercontent.com"
client_secrets_file = os.path.join(pathlib.Path(__file__).parent, "client_secret.json")

flow = Flow.from_client_secrets_file(
    client_secrets_file=client_secrets_file,
    scopes=["https://www.googleapis.com/auth/userinfo.profile", "https://www.googleapis.com/auth/userinfo.email", "openid"],
    redirect_uri="http://127.0.0.1:5000/callback"
)

with open('/home/amir/projects/helloworld/htmlfiles/TeacherFaisal/file.txt', 'r') as file:
    conversation = file.read()


mydb = mysql.connector.connect(
    host='localhost', 
    user='amir', 
    passwd='khoury25.7',
    database='vocabularydb'
)

text = ""
definition = ""
arOutput = ""
mycursor = mydb.cursor()
translator = Translator()
dictionary = PyDictionary()
count = 0
masteredCount = 0
history = []

app = Flask(__name__, static_url_path='/static')
app.secret_key = "AmirKhoury"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/scorePage")
def scorePage():
    return render_template("score.html")

def is_authenticated():
    return "google_id" in session
app.jinja_env.globals.update(is_authenticated=is_authenticated)

def login_is_required(function):
    def wrapper(*args, **kwargs):
        if "google_id" not in session:
            return abort(401)  # Authorization required
        else:
            return function()

    return wrapper


@app.route("/login")
def login():
    authorization_url, state = flow.authorization_url()
    session["state"] = state
    return redirect(authorization_url)


@app.route("/callback")
def callback():
    flow.fetch_token(authorization_response=request.url)



    credentials = flow.credentials
    request_session = requests.session()
    cached_session = cachecontrol.CacheControl(request_session)
    token_request = google.auth.transport.requests.Request(session=cached_session)

    id_info = id_token.verify_oauth2_token(
        id_token=credentials._id_token,
        request=token_request,
        audience=GOOGLE_CLIENT_ID
    )

    session["google_id"] = id_info.get("sub")
    session["name"] = id_info.get("name")
    return redirect("/protected_area")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


@app.route("/loginPage")
def loginPage():
    return render_template("loginPage.html")


@app.route("/protected_area")
@login_is_required
def protected_area():
    if is_authenticated():
        googleAccountName = session['name']
        response = requests.get(f'https://people.googleapis.com/v1/people/{session["google_id"]}?personFields=photos&key={api_key}')
        profile_pic_url = response.json()['photos'][0]['url']
        return render_template("index.html", googleAccname=googleAccountName, profile_pic=profile_pic_url)

@app.route("/base")
def base():
    if is_authenticated():
        googleAccountName = session['name']
        response = requests.get(f'https://people.googleapis.com/v1/people/{session["google_id"]}?personFields=photos&key={api_key}')
        profile_pic_url = response.json()['photos'][0]['url']
        return render_template("base.html", f="ما هي الكلمة التي تريد معرفة معناها" , googleAccname=googleAccountName, profile_pic=profile_pic_url)
    else:
        return redirect('/loginPage')

@app.route("/who-we-are")
def WhoWeAre():
    if is_authenticated():
        googleAccountName = session['name']
        response = requests.get(f'https://people.googleapis.com/v1/people/{session["google_id"]}?personFields=photos&key={api_key}')
        profile_pic_url = response.json()['photos'][0]['url']
        return render_template("who-we-are.html", f="ما هي الكلمة التي تريد معرفة معناها" , googleAccname=googleAccountName, profile_pic=profile_pic_url)
    else:
        return redirect('/loginPage')

        
@app.route("/dictionary")
def mydictionary():
    if is_authenticated():
        googleAccountName = session['name']
        response = requests.get(f'https://people.googleapis.com/v1/people/{session["google_id"]}?personFields=photos&key={api_key}')
        profile_pic_url = response.json()['photos'][0]['url']
        return render_template("dictionary.html", f="ما هي الكلمة التي تريد معرفة معناها" , googleAccname=googleAccountName, profile_pic=profile_pic_url)
    else:
        return redirect('/loginPage')        


@app.route("/dictionary", methods=['POST'])
def my_dictionary_post():
    global masteredCount
    global count
    global history
    global text
    global definition
    global arOutput
    if request.method == "POST":
        text = request.form['text']
        history.append(text)
        definition = dictionary.meaning(text)
        definition_str = str(definition)
        hist_r = str(history).strip('[]')
        Arabic = translator.translate(text, dest="ar")
        arOutput = Arabic.text
        print(arOutput)
        count += 1
        masteredCount +=1
        sqlformula = "INSERT INTO vocabulary (word, definition, arabicDefinition) VALUES (%s, %s, %s)"
        word1 = (text, definition_str, arOutput)
        mycursor.execute(sqlformula , word1)
        mydb.commit()
        return render_template("dictionary.html", d=definition, num=count, f="", hist=hist_r, arOutput = arOutput)
    else:
        return redirect(url_for("dictionary"))

@app.route("/vocabulary")
def vocabulary():
    mycursor.execute('SELECT * FROM vocabulary')
    rows = mycursor.fetchall()
    if is_authenticated():
        googleAccountName = session['name']
        response = requests.get(f'https://people.googleapis.com/v1/people/{session["google_id"]}?personFields=photos&key={api_key}')
        profile_pic_url = response.json()['photos'][0]['url']
        return render_template("vocabularybox.html", word=rows , f="ما هي الكلمة التي تريد معرفة معناها" , googleAccname=googleAccountName, profile_pic=profile_pic_url)
    else:
        return redirect('/loginPage')
    
@app.route('/chatbot')
def chatbot():
        if is_authenticated():
            googleAccountName = session['name']
            response = requests.get(f'https://people.googleapis.com/v1/people/{session["google_id"]}?personFields=photos&key={api_key}')
            profile_pic_url = response.json()['photos'][0]['url']
            return render_template("chatbot.html", f="How Can I help..?" , googleAccname=googleAccountName, profile_pic=profile_pic_url, )
        else:
            return redirect('/loginPage')

@app.route('/get')
def get_bot_response():
    prompt2 = request.args.get('msg')
    prompt = "You are mainly an english teacher named Mr.Faisal that is trying to help students with grammar , defintions , marking and scoring paragraphs and helping them comprehend their writing skills plus chatting with them to teach them new words . allow questions in arabic about english and answer them in arabic , if they try to go of topic tell them Sorry student but I can only help with English " + prompt2
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.5,
)
    bot_response = response.choices[0].text.strip()
    return str(bot_response)


@app.route("/categoriesPage")
def categoriesPage():
    if is_authenticated():
        googleAccountName = session['name']
        response = requests.get(f'https://people.googleapis.com/v1/people/{session["google_id"]}?personFields=photos&key={api_key}')
        profile_pic_url = response.json()['photos'][0]['url']
        return render_template("categoriesPage.html", f="ما هي الكلمة التي تريد معرفة معناها" , googleAccname=googleAccountName, profile_pic=profile_pic_url)
    else:
        return redirect('/loginPage')

@app.route("/readingComprehensionPage")
def readingComprehensionPages():
    if is_authenticated():
        googleAccountName = session['name']
        response = requests.get(f'https://people.googleapis.com/v1/people/{session["google_id"]}?personFields=photos&key={api_key}')
        profile_pic_url = response.json()['photos'][0]['url']

        return render_template("readingComprehensionPage.html", googleAccname=googleAccountName, profile_pic=profile_pic_url)
    else:
        return redirect('/loginPage')

@app.route("/U1ReadingComprehension1")
def U1ReadingComprehension1():
    if is_authenticated():
        googleAccountName = session['name']
        response = requests.get(f'https://people.googleapis.com/v1/people/{session["google_id"]}?personFields=photos&key={api_key}')
        profile_pic_url = response.json()['photos'][0]['url']
        return render_template("U1-ReadingComprehension1.html", googleAccname=googleAccountName, profile_pic=profile_pic_url)
    else:
        return redirect('/loginPage')

@app.route('/checkAnswers' , methods=['POST']  )
def checkAnswers():
    # answer_fields = [
    #     request.args.get("input-field1"),
    #     request.args.get("input-field2"),
    #     request.args.get("input-field3"),
    #     request.args.get("input-field4"),
    #     request.args.get("input-field5"),
    #     request.args.get("input-field6"),
    #     request.args.get("input-field7"),
    #     request.args.get("input-field8"),
    #     request.args.get("input-field9"),
    #     request.args.get("input-field10"),
    #     request.args.get("input-field11"),
    #     request.args.get("input-field12"),
    #     request.args.get("input-field13")
    # ]

    data = request.get_json()
    answers = data['answers']  # Access the 'answers' data

    # Now you can work with 'answers' as a Python list
    for answer in answers:
        print(answer)

    correct_answers = [
        {
            "question": "english Teacher", "content": "You are a teacher comparing the user answer with the correct answer , accept answers that have the same meaning as the correct answer . The correct answer is the area where you feel comfortable / the set of routines and known abilities that make us feel safe",
            "correct_answer": "the area where you feel comfortable / the set of routines and known abilities that make us feel safe"
        },
        {
            "question": "english Teacher", "content": "You are a teacher comparing the user answer with the correct answer , accept answers that have the same meaning as the correct answer . The correct answer is The feeling that they are developing and making progress in their lives",
            "correct_answer": "The feeling that they are developing and making progress in their lives"
        },
        {
            "question": "english Teacher", "content": "You are a teacher comparing the user answer with the correct answer , accept answers that have the same meaning as the correct answer . The correct answer is They may be afraid of failing",
            "correct_answer": "They may be afraid of failing"
        },
        {
            "question": "english Teacher", "content": "You are a teacher comparing the user answer with the correct answer , accept answers that have the same meaning as the correct answer . The correct answer is They are unsure how to begin",
            "correct_answer": "They are unsure how to begin"
        },
        {
            "question": "english Teacher", "content": "You are a teacher comparing the user answer with the correct answer , accept answers that have the same meaning as the correct answer . The correct answer is They make excuses like not wanting to change",
            "correct_answer": "They make excuses like not wanting to change"
        },
        {
            "question": "english Teacher", "content": "You are a teacher comparing the user answer with the correct answer , accept answers that have the same meaning as the correct answer . The correct answer is we can manage",
            "correct_answer": "we can manage"
        },
        {
            "question": "english Teacher", "content": "You are a teacher comparing the user answer with the correct answer , accept answers that have the same meaning as the correct answer . The correct answer is unexpected or worrying",
            "correct_answer": "unexpected or worrying"
        },
        {
            "question": "english Teacher","content": "You are a teacher comparing the user answer with the correct answer , accept answers that have the same meaning as the correct answer . The correct answer is things that are outside our comfort zones",
            "correct_answer": "things that are outside our comfort zones"
        },
        {
            "question": "english Teacher", "content": "You are a teacher comparing the user answer with the correct answer , accept answers that have the same meaning as the correct answer . The correct answer is learning something new",
            "correct_answer": "learning something new"
        },
        {
            "question": "english Teacher", "content": "You are a teacher comparing the user answer with the correct answer , accept answers that have the same meaning as the correct answer . The correct answer is becoming more creative",
            "correct_answer": "becoming more creative"
        },
        {
            "question": "english Teacher", "content": "You are a teacher comparing the user answer with the correct answer , accept answers that have the same meaning as the correct answer . The correct answer is getting fit",
            "correct_answer": "getting fit"
        },
        {
            "question": "english Teacher", "content": "You are a teacher comparing the user answer with the correct answer , accept answers that have the same meaning as the correct answer . The correct answer comfort zone",
            "correct_answer": "comfort zone"
        },
        {
            "question": "english Teacher", "content": "You are a teacher comparing the user answer with the correct answer , accept answers that have the same meaning as the correct answer . The correct answer people",
            "correct_answer": "people"
        }
    ]

    feedback_messages = []

    for i in range(len(correct_answers)):
        answer = answers[i]
        user_answer = correct_answers[i]["correct_answer"]

        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = [
        {"role": "system", "content": "You are an English teacher."},
        {"role": "user", "content": f"The optimal answer is: {answer}"},
        {"role": "user", "content": f"The student's answer is: {user_answer}"},
        {"role": "user", "content": "Please score the student's answer out of 10 based on the optimal answer and provide a numerical score only."},
        ])

        score = re.findall(r'\d+', response.choices[0].message.content)
        print(score)

        feedback_messages = print(response['choices'][0]['message']['content'])
        print(feedback_messages)

    return {"feedback_messages": feedback_messages}

# Now you have the feedback messages and corresponding correct answers in the `feedback_messages` list



@app.route("/U1ReadingComprehension2")
def U1ReadingComprehension2():
    if is_authenticated():
        googleAccountName = session['name']
        response = requests.get(f'https://people.googleapis.com/v1/people/{session["google_id"]}?personFields=photos&key={api_key}')
        profile_pic_url = response.json()['photos'][0]['url']
        return render_template("U1-ReadingComprehension2.html", googleAccname=googleAccountName, profile_pic=profile_pic_url)
    else:
        return redirect('/loginPage')
    
@app.route("/U2ReadingComprehension1")
def U2ReadingComprehension1():
    if is_authenticated():
        googleAccountName = session['name']
        response = requests.get(f'https://people.googleapis.com/v1/people/{session["google_id"]}?personFields=photos&key={api_key}')
        profile_pic_url = response.json()['photos'][0]['url']
        return render_template("U2-ReadingComprehension1.html", googleAccname=googleAccountName, profile_pic=profile_pic_url)
    else:
        return redirect('/loginPage')
    
@app.route("/VocabularyPage")
def VocabularyPage():
    if is_authenticated():
        googleAccountName = session['name']
        response = requests.get(f'https://people.googleapis.com/v1/people/{session["google_id"]}?personFields=photos&key={api_key}')
        profile_pic_url = response.json()['photos'][0]['url']
        return render_template("VocabularyPage.html", googleAccname=googleAccountName, profile_pic=profile_pic_url)
    else:
        return redirect('/loginPage')
    
@app.route("/U1VocabularyQuestion1")
def U1VocabularyQuestion1():
    if is_authenticated():
        googleAccountName = session['name']
        response = requests.get(f'https://people.googleapis.com/v1/people/{session["google_id"]}?personFields=photos&key={api_key}')
        profile_pic_url = response.json()['photos'][0]['url']
        return render_template("U1-VocabularyQuestion1.html", googleAccname=googleAccountName, profile_pic=profile_pic_url)
    else:
        return redirect('/loginPage')
    
@app.route("/U2VocabularyQuestion1")
def U2VocabularyQuestion1():
    if is_authenticated():
        googleAccountName = session['name']
        response = requests.get(f'https://people.googleapis.com/v1/people/{session["google_id"]}?personFields=photos&key={api_key}')
        profile_pic_url = response.json()['photos'][0]['url']
        return render_template("U2-VocabularyQuestion1.html", googleAccname=googleAccountName, profile_pic=profile_pic_url)
    else:
        return redirect('/loginPage')
    
@app.route("/U1VocabularyList1")
def U1VocabularyList():
    if is_authenticated():
        googleAccountName = session['name']
        response = requests.get(f'https://people.googleapis.com/v1/people/{session["google_id"]}?personFields=photos&key={api_key}')
        profile_pic_url = response.json()['photos'][0]['url']
        return render_template("U1-VocabularyList1.html", googleAccname=googleAccountName, profile_pic=profile_pic_url)
    else:
        return redirect('/loginPage')

@app.route("/GrammarPage")
def GrammarPage():
    if is_authenticated():
        googleAccountName = session['name']
        response = requests.get(f'https://people.googleapis.com/v1/people/{session["google_id"]}?personFields=photos&key={api_key}')
        profile_pic_url = response.json()['photos'][0]['url']
        return render_template("GrammarPage.html", googleAccname=googleAccountName, profile_pic=profile_pic_url)
    else:
        return redirect('/loginPage')

@app.route("/U1LanguageExercise1")
def U1LanguageExercise1():
    if is_authenticated():
        googleAccountName = session['name']
        response = requests.get(f'https://people.googleapis.com/v1/people/{session["google_id"]}?personFields=photos&key={api_key}')
        profile_pic_url = response.json()['photos'][0]['url']
        return render_template("U1-LanguageExercise1.html", googleAccname=googleAccountName, profile_pic=profile_pic_url)
    else:
        return redirect('/loginPage')

@app.route("/U1LanguageVideoLesson")
def U1LanguageVideoLesson():
    if is_authenticated():
        googleAccountName = session['name']
        response = requests.get(f'https://people.googleapis.com/v1/people/{session["google_id"]}?personFields=photos&key={api_key}')
        profile_pic_url = response.json()['photos'][0]['url']
        return render_template("U1-LanguageVideoLesson.html", googleAccname=googleAccountName, profile_pic=profile_pic_url)
    else:
        return redirect('/loginPage')

@app.route("/U1LanguageSlideLesson")
def U1LanguageSlideLesson():
    if is_authenticated():
        googleAccountName = session['name']
        response = requests.get(f'https://people.googleapis.com/v1/people/{session["google_id"]}?personFields=photos&key={api_key}')
        profile_pic_url = response.json()['photos'][0]['url']
        return render_template("U1-LanguageSlideLesson.html", googleAccname=googleAccountName, profile_pic=profile_pic_url)
    else:
        return redirect('/loginPage')

@app.route("/WritingPage")
def WritingPage():
    if is_authenticated():
        googleAccountName = session['name']
        response = requests.get(f'https://people.googleapis.com/v1/people/{session["google_id"]}?personFields=photos&key={api_key}')
        profile_pic_url = response.json()['photos'][0]['url']
        return render_template("WritingPage.html", googleAccname=googleAccountName, profile_pic=profile_pic_url)
    else:
        return redirect('/loginPage')

@app.route("/WritingSample1")
def WritingSample1():
    if is_authenticated():
        googleAccountName = session['name']
        response = requests.get(f'https://people.googleapis.com/v1/people/{session["google_id"]}?personFields=photos&key={api_key}')
        profile_pic_url = response.json()['photos'][0]['url']
        return render_template("WritingSample1.html", googleAccname=googleAccountName, profile_pic=profile_pic_url)
    else:
        return redirect('/loginPage')

@app.route("/WritingSample2")
def WritingSample2():
    if is_authenticated():
        googleAccountName = session['name']
        response = requests.get(f'https://people.googleapis.com/v1/people/{session["google_id"]}?personFields=photos&key={api_key}')
        profile_pic_url = response.json()['photos'][0]['url']
        return render_template("WritingSample2.html", googleAccname=googleAccountName, profile_pic=profile_pic_url)
    else:
        return redirect('/loginPage')

@app.route("/WritingSample3")
def WritingSample3():
    if is_authenticated():
        googleAccountName = session['name']
        response = requests.get(f'https://people.googleapis.com/v1/people/{session["google_id"]}?personFields=photos&key={api_key}')
        profile_pic_url = response.json()['photos'][0]['url']
        return render_template("WritingSample3.html", googleAccname=googleAccountName, profile_pic=profile_pic_url)
    else:
        return redirect('/loginPage')

@app.route("/U1WritingQuestion1")
def U1WritingQuestion1():
    if is_authenticated():
        googleAccountName = session['name']
        response = requests.get(f'https://people.googleapis.com/v1/people/{session["google_id"]}?personFields=photos&key={api_key}')
        profile_pic_url = response.json()['photos'][0]['url']
        return render_template("U1-WritingQuestion1.html", googleAccname=googleAccountName, profile_pic=profile_pic_url)
    else:
        return redirect('/loginPage')

@app.route("/SampleTestPage")
def SampleTestPage():
    if is_authenticated():
        googleAccountName = session['name']
        response = requests.get(f'https://people.googleapis.com/v1/people/{session["google_id"]}?personFields=photos&key={api_key}')
        profile_pic_url = response.json()['photos'][0]['url']
        return render_template("SampleTestPage.html", googleAccname=googleAccountName, profile_pic=profile_pic_url)
    else:
        return redirect('/loginPage')

@app.route("/SampleTest1")
def SampleTest1():
    if is_authenticated():
        googleAccountName = session['name']
        response = requests.get(f'https://people.googleapis.com/v1/people/{session["google_id"]}?personFields=photos&key={api_key}')
        profile_pic_url = response.json()['photos'][0]['url']
        return render_template("SampleTest1.html", googleAccname=googleAccountName, profile_pic=profile_pic_url)
    else:
        return redirect('/loginPage')

@app.route("/TawjihiTipsAndTricks")
def TawjihiTipsAndTricks():
    if is_authenticated():
        googleAccountName = session['name']
        response = requests.get(f'https://people.googleapis.com/v1/people/{session["google_id"]}?personFields=photos&key={api_key}')
        profile_pic_url = response.json()['photos'][0]['url']
        return render_template("TawjihiTipsAndTricks.html", googleAccname=googleAccountName, profile_pic=profile_pic_url)
    else:
        return redirect('/loginPage')

@app.route("/U2WritingQuestion1")
def U2WritingQuestion1():
    if is_authenticated():
        googleAccountName = session['name']
        response = requests.get(f'https://people.googleapis.com/v1/people/{session["google_id"]}?personFields=photos&key={api_key}')
        profile_pic_url = response.json()['photos'][0]['url']
        return render_template("U2-WritingQuestion1.html", googleAccname=googleAccountName, profile_pic=profile_pic_url)
    else:
        return redirect('/loginPage')

@app.route("/U3WritingQuestion1")
def U3WritingQuestion1():
    if is_authenticated():
        googleAccountName = session['name']
        response = requests.get(f'https://people.googleapis.com/v1/people/{session["google_id"]}?personFields=photos&key={api_key}')
        profile_pic_url = response.json()['photos'][0]['url']
        return render_template("U3-WritingQuestion1.html", googleAccname=googleAccountName, profile_pic=profile_pic_url)
    else:
        return redirect('/loginPage')

@app.route("/U4WritingQuestion1")
def U4WritingQuestion1():
    if is_authenticated():
        googleAccountName = session['name']
        response = requests.get(f'https://people.googleapis.com/v1/people/{session["google_id"]}?personFields=photos&key={api_key}')
        profile_pic_url = response.json()['photos'][0]['url']
        return render_template("U4-WritingQuestion1.html", googleAccname=googleAccountName, profile_pic=profile_pic_url)
    else:
        return redirect('/loginPage')

@app.route("/U8WritingQuestion1")
def U8WritingQuestion1():
    if is_authenticated():
        googleAccountName = session['name']
        response = requests.get(f'https://people.googleapis.com/v1/people/{session["google_id"]}?personFields=photos&key={api_key}')
        profile_pic_url = response.json()['photos'][0]['url']
        return render_template("U8-WritingQuestion1.html", googleAccname=googleAccountName, profile_pic=profile_pic_url)
    else:
        return redirect('/loginPage')

@app.route("/U9WritingQuestion1")
def U9WritingQuestion1():
    if is_authenticated():
        googleAccountName = session['name']
        response = requests.get(f'https://people.googleapis.com/v1/people/{session["google_id"]}?personFields=photos&key={api_key}')
        profile_pic_url = response.json()['photos'][0]['url']
        return render_template("U9-WritingQuestion1.html", googleAccname=googleAccountName, profile_pic=profile_pic_url)
    else:
        return redirect('/loginPage')

@app.route("/ImageUpload", methods=["POST", "GET"])
def ImageUpload():
    if is_authenticated():
        googleAccountName = session['name']
        response = requests.get(f'https://people.googleapis.com/v1/people/{session["google_id"]}?personFields=photos&key={api_key}')
        profile_pic_url = response.json()['photos'][0]['url']
        
        if request.method == 'POST':
            selected_unit = request.form['Units']

            if request.files:
                image = request.files["image"]
                print(image)

                allowed_extensions = set(['png', 'jpg', 'jpeg', 'pdf'])
                if image.filename.split('.')[-1] not in allowed_extensions:
                    flash('Invalid file type: ' + image.filename)
                    return redirect(request.url)

                file_contents = image.read()
                #print(file_contents)
                encoded_file_contents = base64.b64encode(file_contents)

                FileType = image.filename.split('.')[-1]

                sql = "INSERT INTO ImageAndWord (Unit, File_Type, File) VALUES (%s, %s, %s)"
                val = (selected_unit, FileType, encoded_file_contents)
                mycursor.execute(sql, val)
                mydb.commit()
                last_id = mycursor.lastrowid

                # Fetch image from database
                mycursor.execute("SELECT File, File_Type FROM ImageAndWord")
                fetched = mycursor.fetchall()
                blob = fetched[0][0]  # assuming the image you want is the first one
                file_type = fetched[0][1]

                # Convert BLOB to base64
                base64img = base64.b64encode(blob).decode('utf-8')

                # Create image tag
                img_tag = f'<img src="data:image/{file_type};base64,{encoded_file_contents}" alt="image"/>'
                return render_template('fileDisplay.html', image_id=last_id)
                return render_template("fileDisplay.html", img_tag=img_tag, test=blob)
        
    return render_template("Imageupload1.html", base64 = base64, googleAccname=googleAccountName, profile_pic=profile_pic_url)


@app.route('/image/<int:image_id>')
def serve_image(image_id):
    # Fetch the image data from the database
    mycursor.execute("SELECT File, File_Type FROM ImageAndWord WHERE id = %s", (image_id,))
    fetched = mycursor.fetchone()
    if fetched is None:
        # No such image
        abort(404)
    img_data = fetched[0]
    file_type = fetched[1]

    # Create a BytesIO object and send it as a file
    
    return send_file(BytesIO(img_data), mimetype=f'image/{file_type}')

@app.route('/fileDisplay')
def fileDisplay():
    return render_template("fileDisplay.html")

if __name__ == '__main__':
    app.run(debug=True)