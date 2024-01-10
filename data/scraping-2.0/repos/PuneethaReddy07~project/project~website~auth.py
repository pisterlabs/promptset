import csv
from flask_login import current_user
from sentence_transformers import util,SentenceTransformer
from io import TextIOWrapper
import time
import jwt
from datetime import datetime, timedelta
from flask import Blueprint, render_template, request, flash, session, redirect, url_for
from .models import TestResultData, User,TestResult,TopicResult
from . import mongo2
from werkzeug.security import generate_password_hash, check_password_hash
from flask import current_app
import openai


auth = Blueprint('auth', __name__)

def get_random_questions(num_questions,sub):
    if(sub=='java'):

        questions = mongo2.db.java.aggregate([
            { '$sample': { 'size': num_questions } },
            { '$project': { '_id': 0, 'Question': 1 } }
        ])
        return list(questions)
    else:
        questions = mongo2.db.python.aggregate([
            { '$sample': { 'size': num_questions } },
            { '$project': { '_id': 0, 'Question': 1 } }
        ])
        return list(questions)



@auth.route('/java')
def java():
    num_questions = 10  # Specify the number of questions needed
    questions = get_random_questions(num_questions,sub='java')
    
    return render_template('java.html', questions=questions,num_questions=num_questions)

@auth.route('/python')
def python():
    num_questions = 10  # Specify the number of questions needed
    questions = get_random_questions(num_questions,sub='python')
    
    return render_template('python.html', questions=questions,num_questions=num_questions)

@auth.route('/')
def home():
    token = session.get('token')
    if token:
        try:
            payload = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])
            user_id = payload['user_id']
            user = User.objects(id=user_id).first()
            if user:
                last_activity = session.get('last_activity')
                if last_activity and time.time() - last_activity > 24 * 60 * 60:  # one day of inactivity
                    session.pop('token', None)  # Remove the token from the session
                    flash('Session expired. Please log in again.')
                    return redirect(url_for('auth.login'))
                return render_template("userdash.html", user=user)
            else:
                flash('Invalid token')
        except jwt.ExpiredSignatureError:
            flash('Token expired')
            return redirect(url_for('auth.logout'))
        except jwt.InvalidTokenError:
            flash('Invalid token')
            return redirect(url_for('auth.logout'))

    return redirect(url_for('auth.login'))


@auth.route("/uploadpage")
def uploadpage():
    return render_template("upload.html")


ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@auth.route('/upload', methods=['GET', 'POST'])
def upload():
    uploaded_count = 0
    skipped_count = 0

    if request.method == 'POST':
        subject = request.form.get('subject')
        csv_file = request.files.get('csv_file')

        if csv_file and allowed_file(csv_file.filename):
            
            csv_data = TextIOWrapper(csv_file, encoding='utf-8')
            reader = csv.DictReader(csv_data)

            for row in reader:
                Question = row.get('Question')
                if not Question:
                    skipped_count += 1
                    continue  
                existing_record = mongo2.db[subject].find_one({'Question': Question})
                if existing_record is not None:
                    skipped_count += 1
                    continue 

                else:
                    mongo2.db[subject].insert_one(row)
                    uploaded_count += 1

    return render_template('upload.html', uploaded_count=uploaded_count, skipped_count=skipped_count)



@auth.route('/evaluate', methods=['POST'])
def evaluate_answers():
    token = session.get('token')
    payload = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])
    user_id = payload['user_id']
    user = User.objects(id=user_id).first()
    model_name = 'paraphrase-MiniLM-L6-v2'
    sentence_transformer = SentenceTransformer(model_name)
    subject = request.form.get('subject')
    user_answers = request.form.getlist('answer')
    questions = request.form.getlist('question')

    difficulty_dict = {}
    question_answers = {}
    question_topic={}

    for question in questions:
        doc = mongo2.db[subject].find_one({'Question': question})
        topic=doc['\ufeffTopic']
        difficulty = doc['Difficulty']
        answer = doc['Answer']
        difficulty_dict[question] = difficulty
        question_answers[question] = answer
        question_topic[question]=topic
          
    # Pre-encode sentence embeddings for all answers
    answer_embeddings = sentence_transformer.encode(list(question_answers.values()))

    difficulty_counts = {'Easy': {'Right': 0, 'Wrong': 0}, 'Medium': {'Right': 0, 'Wrong': 0}, 'Hard': {'Right': 0, 'Wrong': 0}}
    results = []
    correct = 0
 
    easy_topics = []
    medium_topics = []
    hard_topics = []
    for question, user_answer, answer_embedding in zip(questions, user_answers, answer_embeddings):
        difficulty = difficulty_dict[question]
        actual_embedding = answer_embedding

        user_embedding = sentence_transformer.encode([user_answer])[0]

        similarity = util.cos_sim(user_embedding, actual_embedding)
        percentage = float(similarity * 100)

        if percentage >= 70:
            correctness = 'Right'
            correct += 1
            difficulty_counts[difficulty]['Right'] += 1

            # Update counts for topics
            topic = question_topic[question]
        if percentage >= 70:
            correctness = 'Right'
            correct += 1
            difficulty_counts[difficulty]['Right'] += 1

            # Update counts for topics
            topic = question_topic[question]
            if difficulty == 'Easy':
                update_topic_counts(easy_topics, topic, True)
            elif difficulty == 'Medium':
                update_topic_counts(medium_topics, topic, True)
            elif difficulty == 'Hard':
                update_topic_counts(hard_topics, topic, True)
        else:
            correctness = 'Wrong'
            difficulty_counts[difficulty]['Wrong'] += 1

            # Update counts for topics
            topic = question_topic[question]
            if difficulty == 'Easy':
                update_topic_counts(easy_topics, topic, False)
            elif difficulty == 'Medium':
                update_topic_counts(medium_topics, topic, False)
            elif difficulty == 'Hard':
                update_topic_counts(hard_topics, topic, False)

        result = {
            'question': question,
            'correctness': correctness
        }
        results.append(result)

    total_questions = len(questions)
    correct_easy_questions = sum(1 for r in results if r['correctness'] == 'Right' and difficulty_dict[r['question']] == 'Easy')
    correct_medium_questions = sum(1 for r in results if r['correctness'] == 'Right' and difficulty_dict[r['question']] == 'Medium')
    correct_hard_questions = sum(1 for r in results if r['correctness'] == 'Right' and difficulty_dict[r['question']] == 'Hard')

    total_easy_questions = sum(1 for difficulty in difficulty_dict.values() if difficulty == 'Easy')
    total_medium_questions = sum(1 for difficulty in difficulty_dict.values() if difficulty == 'Medium')
    total_hard_questions = sum(1 for difficulty in difficulty_dict.values() if difficulty == 'Hard')
    
        # Create a new test result object
    save_test_result(user.email, subject, correct, total_questions,
                     total_easy_questions, total_medium_questions, total_hard_questions,
                     correct_easy_questions, correct_medium_questions, correct_hard_questions,
                     easy_topics, medium_topics, hard_topics)
    return render_template('evaluate.html', correct=correct, total_questions=total_questions,
                           total_easy_questions=total_easy_questions, total_medium_questions=total_medium_questions,
                           total_hard_questions=total_hard_questions, correct_easy_questions=correct_easy_questions,
                           correct_medium_questions=correct_medium_questions,
                           correct_hard_questions=correct_hard_questions)



def update_topic_counts(topic_results, topic, is_correct):
    # Access the list within the ListField object
    topic_results_list = topic_results if isinstance(topic_results, list) else []

    # Check if the topic already exists in the list
    for topic_result in topic_results_list:
        if topic_result.topic == topic:
            if is_correct:
                topic_result.correct_answers += 1
            topic_result.number_of_questions += 1
            return

    # If the topic doesn't exist, create a new topic result object
    topic_result = TopicResult(topic=topic, correct_answers=1 if is_correct else 0, number_of_questions=1)
    topic_results_list.append(topic_result)

    # Save the updated list back to the ListField object if necessary
    if not isinstance(topic_results, list):
        topic_results = topic_results_list

    



def save_test_result(email, subject, correct, total_questions,
                     total_easy_questions, total_medium_questions, total_hard_questions,
                     correct_easy_questions, correct_medium_questions, correct_hard_questions,
                     easy_topics, medium_topics, hard_topics):
    def existing_test_results():
        existing_test_results = TestResultData.objects(email=email).first()

        if existing_test_results:
            return len(existing_test_results.test_results) + 1
        else:
            return 1

    test_result = TestResult(
        subject=subject,
        test_number=str(existing_test_results()),
        correct=correct,
        total_questions=total_questions,
        total_easy_questions=total_easy_questions,
        total_medium_questions=total_medium_questions,
        total_hard_questions=total_hard_questions,
        correct_easy_questions=correct_easy_questions,
        correct_medium_questions=correct_medium_questions,
        correct_hard_questions=correct_hard_questions,
        date=datetime.now().strftime("%Y-%m-%d"),
        time=datetime.now().strftime("%H:%M:%S"),
        easy_topics=easy_topics,
        medium_topics=medium_topics,
        hard_topics=hard_topics
    )

    existing_test_results = TestResultData.objects(email=email).first()

    if existing_test_results:
        existing_test_results.test_results.append(test_result)
        existing_test_results.save()
    else:
        test_results_data = TestResultData(email=email, test_results=[test_result])
        test_results_data.save()

    



@auth.route('/result', methods=["POST", "GET"])
def result():
    token = session.get('token')
    payload = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])
    user_id = payload['user_id']
    user = User.objects(id=user_id).first()
    email = user.email

    # Retrieve the test results from the database
    test_results_data = TestResultData.objects(email=email).first()
    test_results = test_results_data.test_results if test_results_data else []

    return render_template("result.html", test_results=test_results)




@auth.route('/login', methods=["GET", "POST"])
def login():
    if 'token' in session:
        return redirect(url_for('auth.home'))
    elif request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.objects(email=email).first()
        if user:
            if check_password_hash(user.password, password):
                flash('Logged in successfully')
                token = jwt.encode(
                    {
                        'user_id': str(user.id),
                        'email': user.email,
                        'role': user.role,
                        'exp': datetime.utcnow() + timedelta(days=1)  
                    },
                    current_app.config['SECRET_KEY'],  
                    algorithm='HS256'
                )
                session['token'] = token 
                return redirect(url_for('auth.home'))
            else:
                flash('Incorrect password')
        else:
            flash('Email does not exist')

    return render_template("login.html")




@auth.route("/profile")
def profile():
    token=session.get('token')
    payload = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])
    user_id = payload['user_id']
    user = User.objects(id=user_id).first()
    return render_template('profile.html',user=user)

@auth.route('/logout')
def logout():
    session.pop('token', None)
    return redirect(url_for('auth.login'))

@auth.route('/signup', methods=["GET", "POST"])
def signup():
    if 'user_id' in session:
        return redirect(url_for('views.home'))
    elif request.method == 'POST':
        email = request.form.get("email")
        firstName = request.form.get("firstName")
        password1 = request.form.get("password1")
        password2 = request.form.get("password2")
        user = User.objects(email=email).first()
        if user:
            flash('Email already exists')
        elif len(email) < 4:
            flash('Invalid email')
        elif len(firstName) < 4:
            flash('Username too short')
        elif len(password1) < 6:
            flash('Password is too short')
        elif password2 != password1:
            flash("Passwords don't match")
        else:
            new_user = User(
                email=email,
                first_name=firstName,
                password=generate_password_hash(password1, method='sha256'),
                role='admin' if email == 'admin@gmail.com' else 'student'
            )
            new_user.save()
            flash('Sign up successful')
            return redirect(url_for('auth.home'))

    return render_template("signup.html")
@auth.route('/ask',methods=["GET", "POST"])
def ask():
    return render_template("takeq.html")

openai.api_key ="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"
messages = [{"role": "system", "content": "You are a computer science expert"}]
def CustomChatGPT(user_input):
    messages.append({"role": "user", "content": user_input})
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = messages
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": ChatGPT_reply})
    return ChatGPT_reply

@auth.route('/ans', methods=['GET', 'POST'])
def ans():
    if request.method == 'POST':
        user_input = request.form['query']
        response=CustomChatGPT(user_input)
        # # Call ChatGPT API
        # response = openai.Completion.create(
        #     engine='davinci',
        #     prompt=user_input,
        #     max_tokens=100
        # )

        # # Extract the generated response from the API
        # chatgpt_response = response.choices[0].text.strip()

        # # Split the response by sentences and select the first sentence
        # sentences = re.split(r'(?<=[.!?])\s+', chatgpt_response)
        # if len(sentences) > 0:
        #     chatgpt_response = sentences[0]

        return render_template('answer.html', response=response)
    else:
        return render_template('takeq.html')

