from app.auth import Authentication
from app.openai_generate import OpenAIService
from app import app
from pydantic import BaseModel
from flask import render_template, request, redirect, url_for, flash, request, jsonify
from flask_login import login_required, current_user

auth = Authentication()


class Data(BaseModel):
    """ Data class for data validation """
    keyword: str


@app.route("/api/compose", methods=['POST'])
def compose_article():
    data = request.get_json()
    print(data)
    keyword = data['keyword']

    openai_service = OpenAIService()

    # generate article using the OpenAI API
    article = openai_service.generate_article(keyword)

    return jsonify({'article': article})


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if auth.login(username, password):
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'error')
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        if auth.signup(username, email, password):
            return redirect(url_for('home'))
    return render_template('signup.html')


@app.route('/home')
@login_required
def home():
    return render_template('home.html', username=current_user.username)


@app.route('/logout')
@login_required
def logout():
    auth.logout()
    return redirect(url_for('login'))


@app.route('/reset_password', methods=['GET', 'POST'])
@login_required
def reset_password():
    if request.method == 'POST':
        new_password = request.form['new_password']
        auth.reset_password(current_user, new_password)
        flash('Password reset successful', 'success')
        return redirect(url_for('home'))
    return render_template('reset_password.html')
