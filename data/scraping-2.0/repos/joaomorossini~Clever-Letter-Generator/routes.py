# External dependencies
import openai
import io
import os
import tempfile
from datetime import datetime
from flask import render_template, request, url_for, redirect, flash, Response, session, send_file, Markup
from flask_login import login_user, login_required, logout_user, current_user
from flask_mail import Message
# Internal dependencies
from models import User, Log
from forms import SignupForm, LoginForm, RequestResetForm, ResetPasswordForm
from app import app, db, bcrypt, mail, login_manager, limiter
from prompt_template import prompt_template


# Security measures for the Heroku production environment
@app.before_request
def enforce_https():
    if request.headers.get('X-Forwarded-Proto') == 'http' and not app.debug:
        request_url = request.url.replace('http://', 'https://', 1)
        return redirect(request_url, code=301)

@app.after_request
def set_hsts_header(response):
    if request.url.startswith('https://'):
        response.headers['Strict-Transport-Security'] = 'max-age=31536000'  # One year
    return response


@login_manager.user_loader
@limiter.limit("10/minute")
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route('/cleverletter/', methods=['GET', 'POST'])
@limiter.limit("10/minute")
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('generator'))
    form = SignupForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html', title='Sign Up', form=form)


@app.route('/cleverletter/login', methods=['GET', 'POST'])
@limiter.limit("10/minute")
def login():
    if current_user.is_authenticated:
        return redirect(url_for('generator'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            return redirect(url_for('dashboard'))
        else:
            flash('Login Unsuccessful. Please make sure you used the correct credentials', 'warning')
    return render_template('login.html', form=form)


@app.route('/cleverletter/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/cleverletter/generator', methods=['GET', 'POST'])
@limiter.limit("5/minute")
def generator():
    user_authenticated = current_user.is_authenticated
    response = ""
    job_title = ""
    job_description = ""
    employer_name = ""
    employer_description = ""
    additional_instructions = ""

    if not current_user.is_authenticated:
        flash(Markup('<a href="{}">Sign up</a> or <a href="{}">Login</a> to keep your CV and cover letter history'.format(url_for('signup'), url_for('login'))), 'warning')

    # Retrieve CV from session for unauthenticated users or from the database for authenticated users
    if current_user.is_authenticated and current_user.cv:
        cv = current_user.cv
    else:
        cv = session.get('cv', "Your CV goes here")

    if request.method == 'POST':
        job_title = request.form.get('job_title')
        job_description = request.form.get('job_description')
        employer_name = request.form.get('employer_name')
        employer_description = request.form.get('employer_description')
        additional_instructions = request.form.get('additional_instructions')
        session_cv = request.form.get('cv')  # Assuming the CV is submitted as a form field

        # Update CV in session for unauthenticated users
        if not current_user.is_authenticated and session_cv:
            session['cv'] = session_cv
            cv = session_cv

        if 'generate' in request.form:
            if cv == "Your CV goes here":
                flash('Please set your CV before generating a cover letter.', 'warning')
                return render_template('dashboard.html', job_title=job_title, job_description=job_description,
                                       employer_name=employer_name, employer_description=employer_description,
                                       additional_instructions=additional_instructions)

            prompt = prompt_template.format(cv=cv, job_title=job_title, job_description=job_description,
                                            employer_name=employer_name, employer_description=employer_description,
                                            additional_instructions=additional_instructions)
            try:
                response = get_completion(prompt)
            except Exception as e:
                flash('Error generating cover letter: {}'.format(str(e)), 'error')
                return redirect(url_for('generator'))

            # Save the response in the user's session
            session['response'] = response

            # Create a log entry only for authenticated users
            if current_user.is_authenticated:
                log = Log(job_title=job_title, employer_name=employer_name, user_id=current_user.id)
                db.session.add(log)
                try:
                    db.session.commit()
                except Exception as e:
                    flash('Error saving log: {}'.format(str(e)), 'error')
                    return redirect(url_for('generator'))

            # Save the response to a txt file in a temporary directory
            filename = '{} - {} - {}.txt'.format(employer_name, job_title, datetime.now().strftime('%d-%b-%Y'))
            temp_dir = tempfile.gettempdir()
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, 'w') as f:
                f.write(response)
            # Save the filename in the session
            session['filename'] = file_path

        elif 'clear' in request.form:
            job_title = ""
            job_description = ""
            employer_name = ""
            employer_description = ""
            additional_instructions = ""
            session['response'] = ""

        elif 'download' in request.form:
            # Get the filename from the session
            file_path = session.get('filename')
            if file_path and os.path.exists(file_path):
                download_response = send_file(file_path, as_attachment=True)
                os.remove(file_path)  # delete the file after sending it
                return download_response
            else:
                flash('No cover letter available for download.', 'warning')
                return redirect(url_for('generator'))

    return render_template('generator.html', response=response, job_title=job_title, job_description=job_description,
                           employer_name=employer_name, employer_description=employer_description,
                           additional_instructions=additional_instructions, cv=cv, user_authenticated=user_authenticated, user=current_user)


@app.route('/cleverletter/dashboard', methods=['GET', 'POST'])
@limiter.limit("5/minute")
def dashboard():
    # Initialize CV with a default value
    cv = "Your CV goes here"
    logs = None
    user_authenticated = current_user.is_authenticated

    if request.method == 'POST':
        # Handle CV form submission
        new_cv = request.form.get('cv')
        if new_cv:
            if current_user.is_authenticated:
                current_user.cv = new_cv
                db.session.commit()
                flash('CV updated successfully.', 'success')
            else:
                session['cv'] = new_cv
                flash('CV saved to session successfully.', 'success')

    # Fetch CV from the authenticated user or from the session
    if current_user.is_authenticated:
        cv = current_user.cv if current_user.cv else cv
        # Fetch the logs from the database
        page = request.args.get('page', 1, type=int)
        per_page = 10
        logs = Log.query.filter_by(user_id=current_user.id).order_by(Log.timestamp.desc()).paginate(page=page, per_page=per_page)
    else:
        cv = session.get('cv', cv) # Use the session value if available, otherwise use the default

    return render_template('dashboard.html', user_authenticated=user_authenticated, user=current_user, cv=cv, logs=logs)


@app.route('/cleverletter/reset_request', methods=['GET', 'POST'])
@limiter.limit("5/minute")
def reset_request():
    form = RequestResetForm()
    message = None
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user:
            send_reset_email(user)
            message = 'An e-mail has been sent with instructions to reset your password.'
    return render_template('reset_request.html', form=form, message=message)


def send_reset_email(user):
    token = user.get_reset_token()
    msg = Message('Password Reset Request',
                  sender='noreply@demo.com',
                  recipients=[user.email])
    msg.body = f'''To reset your password, visit the following link:
{url_for('reset_token', token=token, _external=True)}
If you did not make this request then simply ignore this email and no changes will be made.
'''
    mail.send(msg)


@app.route('/cleverletter/reset_request/<token>', methods=['GET', 'POST'])
@limiter.limit("5/minute")
def reset_token(token):
    user = User.verify_reset_token(token)
    if not user:
        # If the token is invalid or expired, redirect the user to the `reset_request` route.
        return redirect(url_for('reset_request'))
    form = ResetPasswordForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        user.password = hashed_password
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('reset_token.html', form=form)


@app.route('/cleverletter/delete_account', methods=['POST'])
@limiter.limit("5/minute")
@login_required
def delete_account():
    user = User.query.get(current_user.id)
    db.session.delete(user)
    db.session.commit()
    flash('Your account has been deleted.', 'success')
    return redirect(url_for('signup'))


def get_completion(prompt, model="gpt-3.5-turbo"):
    # Always use the development API key
    api_key = app.config['OPENAI_API_KEY_DEV']

    # Set the API key for this request
    openai.api_key = api_key

    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.5,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]