from flask import Blueprint, render_template, url_for, redirect, request, flash, jsonify, session
from flask_login import current_user, login_user, logout_user, login_required
from sqlalchemy import func, or_, and_
from datetime import datetime
import openai
import spacy
import re
import string
import secrets

from .models import User, UserSession, UserQuestionAnswer, Question
from .app import db, login_manager
openai.api_key = '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'


main_bp = Blueprint('main_bp', __name__)

@main_bp.route('/')
def index():
    print('Index route called')
    return render_template('index.html')

@main_bp.route('/login', methods=['GET', 'POST'])
def login():
    print("Login route called")
    if request.method == 'POST':
        print('Login route - POST method')
        username = request.form['login-username']
        password = request.form['login-password']
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            session['user_id'] = user.id
            session_token = secrets.token_hex(32)  # Generate a random token
            session['token'] = session_token  # Store the token in session
            new_session = UserSession(user_id=user.id, token=session_token)  # Pass the token when creating a UserSession
            db.session.add(new_session)
            db.session.commit()
            return jsonify(status="success", message="Login successful."), 200

        else:
            print('Login failed. Check your username and/or password.')
            return jsonify(status="error", message="Login failed. Check your username and/or password."), 400

    return redirect(url_for('chat'))


@main_bp.route('/register', methods=['GET', 'POST'])
def register():
    print("Register route - start")
    if current_user.is_authenticated:
        print("Register route - user already authenticated")
        return redirect(url_for('index'))

    if request.method == 'POST':
        print("Register route - POST method")
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        email = request.form['email']

        print(
            f"Received form data: username={username}, email={email}, password={confirm_password}")
        
        if password < 8:
            print('Password must be at least 8 characters long.')
            return jsonify(status="error", message="Password must be at least 8 characters long."), 400
        
        if username < 6:
            print('Username must be at least 6 characters long.')
            return jsonify(status="error", message="Username must be at least 6 characters long."), 400

        if password != confirm_password:
            print('Passwords do not match.')
            return jsonify(status="error", message="Passwords do not match."), 400

        existing_user = User.query.filter_by(username=username).first()
        if existing_user is not None:
            print('Username already exists.')   
            return jsonify(status="error", message="Username already exists."), 400

        # Check for empty password
        if not password:
            return jsonify(status="error", message="Password cannot be empty."), 400

        # Check for valid email format
        email_regex = r"[^@]+@[^@]+\.[^@]+"
        if not re.match(email_regex, email):
            return jsonify(status="error", message="Invalid email format."), 400

        # Check for long username
        if len(username) > 64:
            return jsonify(status="error", message="Username cannot be more than 64 characters."), 400

        # Check for non-printable username
        if not all(c in string.printable for c in username):
            return jsonify(status="error", message="Username cannot contain non-printable characters."), 400

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        print("User registered")
        return jsonify(status="success", message="User registered."), 200
    return render_template('index.html')


@main_bp.route('/logout')
def logout():
    try:
        user_session = UserSession.query.filter_by(
            user_id=session['user_id'], token=session['token'], logout_time=None).first()
        if user_session:
            user_session.logout_time = datetime.now()  # Record the logout time
            db.session.commit()

            # Calculate the session duration in seconds
            user_session.duration = (
                user_session.logout_time - user_session.login_time).total_seconds()
            db.session.commit()
    except Exception as e:
        print(f"An error occurred during logout: {e}")
    finally:
        session.pop('token', None)  # Remove the session token
        logout_user()
        flash('You have been logged out.', category='success')
        return render_template('index.html')


@main_bp.route('/users')
@login_required
def users():
    # Renders the users page with all the users.
    # Used for debugging purposes.
    all_users = User.query.all()
    return render_template('users.html', users=all_users)


@main_bp.route('/about')
def about():
    # Renders the about page with project information.
    return render_template('about.html')


@main_bp.route('/chatbot')
@login_required
def chatbot_login():
    return render_template('chatbot.html')


@main_bp.route("/start", methods=["GET"])
@login_required
def start_chat():
    session["state"] = "get_name"  # set the initial state
    response = "Hi, I'm Alice. At Reli-AI we understand that dating is complicated.\n What is your name?"
    return jsonify({"message": response})


@main_bp.route("/history", methods=["GET", "POST"])
@login_required
def history():
    search_term = request.args.get('q')

    # Fetch the questions
    if search_term:
        questions = Question.query.filter(Question.question_text.contains(search_term)).all()
    else:
        questions = Question.query.all()

    # Fetch the user's answers to the questions
    if search_term:
        user_question_answers = UserQuestionAnswer.query.filter(
            and_(
                UserQuestionAnswer.user_id == current_user.id,
                or_(
                    Question.question_text.contains(search_term),
                    UserQuestionAnswer.submitted_answer.contains(search_term)
                )
            )
        ).all()
    else:
        user_question_answers = UserQuestionAnswer.query.filter_by(user_id=current_user.id).all()
    # This line was added to sort the answers by question id
    return render_template("history.html", questions=questions, user_question_answers=user_question_answers, search_term=search_term, zip=zip)
 
@main_bp.route("/history/delete/<int:answer_id>", methods=["GET", "POST"])
@login_required
def delete_answer(answer_id):
    answer = UserQuestionAnswer.query.get(answer_id)
    if answer and answer.user_id == current_user.id:
        db.session.delete(answer)
        db.session.commit()
        return redirect(url_for('history'))
    else:
        return jsonify({"error": "Answer not found or user not authorized"}), 404



@main_bp.route("/chat", methods=["POST"])
@login_required
def chat():
    user_message = request.json["message"].strip().capitalize()
    response = ""
    buttons = []

    if session.get("state") == "get_name":
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(user_message)
        name = None
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text
                break
        if name:
            session["name"] = name
            session["saved_name"] = name
            session["state"] = "get_age"
            response = f"Hi {name}, are you over 18 years old? (yes/no)"
            
        else:
            session["name"] = user_message
            session["saved_name"] = user_message
            response = f"Hi {user_message}, What a name :), are you over 18 years old? (yes/no)"
            session["state"] = "get_age"
            

    elif session.get("state") == "get_age":
        if user_message.lower().strip() == "yes":
            session["state"] = "get_age_value"
            response = "Great! How old/young are you?"
        elif user_message.lower().strip() == "no":
            session.clear()
            response = "Sorry, you must be over 18 to use this service"
        else:
            response = "Please answer with 'yes' or 'no'"

    elif session.get("state") == "get_age_value":
        try:
            age = int(user_message)
            if age >= 18:
                session["age"] = age
                session["saved_age"] = age
                session["state"] = "permission_to_use"
                response = f"Thanks {session['saved_name']}, we at Reli AI understand that life is both too short, and too long to go without someone just for you. May we help you? (yes/no)"

            else:
                session["age"] = age
                session["saved_age"] = age
                response = f"Sorry, {session['saved_name']} you must be over 18 to use this service :D"
                session["state"] = "session_end"
        except ValueError:
            response = "Please enter a valid age"

    
    elif session.get("state") == "permission_to_use":
        if user_message.lower().strip() == "yes":
            response = f"Wonderful {session['saved_name']}, we will now ask you a series of questions designed to help us understand more about you, and may help you understand more about yourself. Shall we begin? (yes/no)"
            session["state"] = "question_1"
        elif user_message.lower().strip() == "no":
            response = "Maybe another time :)"
            session["state"] = "session_end"
        else:
            response = "Please answer with 'yes' or 'no'"

    elif session.get("state") and session.get("state").startswith("question_"):
        question_number = int(session.get("state").split("_")[1])
        question = Question.query.get(question_number)

        if question:
            if current_user.is_authenticated:
                # Save the user's answer to the database
                user_question_answer = UserQuestionAnswer(user=current_user, question=question, submitted_answer=user_message)
                db.session.add(user_question_answer)
                db.session.commit()
            else:
                response = "You must be logged in to answer questions."
                session["state"] = "session_end"
                print("User not logged in.")
                return jsonify({"message": response}), 400

            # Fetch the next question
            next_question = Question.query.get(question_number + 1)

            if next_question:
                # Ask the user the next question
                response = next_question.question_text
                session["state"] = f"question_{question_number + 1}"
            else:
                response = "Thank you for answering all the questions!\n I will now generate a unique dating profile for you based on your answers."
                session["state"] = "session_end"

                # call openai api
                response_text = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=generate_prompt(),
                    max_tokens=300,
                    temperature=0.5
                )
                response = response_text["choices"][0]["text"]
                
                # Create a new question if it doesn't exist
                question = Question.query.filter_by(question_text='unique dating profile').first()
                if question is None:
                    question = Question(question_text='unique dating profile', question_type='type1')
                    db.session.add(question)
                    db.session.commit()  # commit needed to assign an ID to the new question

                # Create a new UserQuestionAnswer
                user_question_answer = UserQuestionAnswer(
                    user_id=current_user.id,
                    question_id=question.id,  # use the id of the question
                    submitted_answer=response
                )

                # Add and commit the new UserQuestionAnswer to the database
                db.session.add(user_question_answer)
                db.session.commit()

                # maybe you want to do something with the result
                # here is a print to debug the returned text
                print(response_text)
        else:
            # The current question doesn't exist
            response = "An error occurred. Please try again."
            session["state"] = "session_end"
            print("debug 5 - question does not exist")

    return jsonify({"message": response, "buttons": buttons})

def generate_prompt():
    user_question_answers = UserQuestionAnswer.query.filter_by(user_id=current_user.id).all()

    # Build the prompt with all the questions and answers
    prompt = "I am an AI trained to help create engaging and interesting dating profiles. Here is the information I've gathered about a person:\n\n"
    for uqa in user_question_answers:
        prompt += f"- {uqa.question.question_text}: {uqa.submitted_answer}\n"

    prompt += "\nBased on this information, please generate an attractive and compelling dating profile for this person. insert the information required."

    return prompt
