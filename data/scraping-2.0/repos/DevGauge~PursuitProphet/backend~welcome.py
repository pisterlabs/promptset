import os
import sys
from flask import Flask, jsonify, render_template, request, redirect, url_for, flash, session, after_this_request
sys.path.insert(0, '../')
sys.path.insert(0, '/app')
# from app.models import ChatBot
from langchain_m.langchain_module import TaskChatBot, GoalChatBot
import app.app
from app.models import Goal, Task, User
from app.app import app_instance
from app.pp_logging.db_logger import db
from flask_socketio import SocketIO, send, emit
from flask_security import roles_required, login_required, login_user, user_registered, current_user
from flask_security.confirmable import confirm_user, confirm_email_token_status
from LangChainAgentFactory import AgentFactory
from urllib.parse import quote
# import feature blueprints
import features.demo.demo_blueprint
import features.dream.dream_blueprint
import features.task.task_blueprint
import features.subtask.subtask_blueprint
import features.google_oauth.google_blueprint
import features.profile.profile_blueprint

from shared.blueprints.blueprints import register_blueprints
app = app_instance.app
socketio = SocketIO(app, debug=True)
register_blueprints(app)
print(app.url_map)

@user_registered.connect_via(app)
def user_registered_sighandler(sender, user, confirm_token, confirmation_token, form_data):
    @after_this_request
    def transfer_goals_after_request(response):
        if 'temp_user_id' in session:
            temp_user_id = session['temp_user_id']
            temp_user = User.query.get(temp_user_id)
            if temp_user is not None:
                # Transfer the goals from the temporary user to the registered user
                for goal in temp_user.goals:
                    user.goals.append(goal)
            
                # Ensure that goals are persisted in the database before deleting temp_user
                db.session.flush()

                # Delete the temporary user
                db.session.delete(temp_user)
                db.session.commit()
        return response

@app.teardown_appcontext
def shutdown_session(exception=None):
    db.session.remove()

@socketio.on('message')
def handle_message(data):
    print('received message: ' + data)
    send(data, broadcast=True)

from sqlalchemy import exc
@app.errorhandler(exc.SQLAlchemyError)
def handle_db_exceptions(error):
    db.session.rollback()

@app.route('/error/<string:error_message>')
def error_page(error_message):
    print(f'error message: {error_message}')
    # traceback.print_exc()
    return render_template('error.html', error_message=error_message, pageTitle='Error')

@app.errorhandler(500)
def handle_500(error):
    error_message = "Internal Server Error"
    return redirect(url_for('error_page', error_message=error_message))

@app.route('/admin')
@login_required
@roles_required('admin')
def admin_home():
    return "Hello, Admin!"

@app.route('/thank_you')
def thank_you():
    return render_template('thank_you.html')

@app.route('/confirm/<token>')
def confirm_email(token):
    # Check the token status first
    expired, invalid, user = confirm_email_token_status(token)
    
    if not expired and not invalid:
        # confirm the user
        if confirm_user(token):
            # if successful, log the user in
            login_user(user)
            return redirect(url_for('dashboard'))
    return 'The confirmation link is invalid or has expired.'

@app.route('/feature/<feature_key>', methods=['GET'])
def get_user_feature(feature_key):
    user_id = request.args.get('user_id', None)
    if user_id is None:
        user = current_user
    else:
        user = User.query.get(user_id)

    feature_value = getattr(user, feature_key, None)
    if feature_value is not None:
        print(f'feature key value: {feature_value}')
        return jsonify({feature_key: feature_value})
    else:
        print(f'feature key {feature_key} not found. user attribs: {user.__dict__}')
        return jsonify({feature_key: False})
    
@app.route('/feature/<feature_key>', methods=['PUT'])
def update_feature_value(feature_key):
    user_id = request.args.get('user_id', None)
    print(f'user id from request: {user_id}')
    if user_id is None:
        user = current_user
    else:
        user = User.query.get(user_id)
    if hasattr(user, feature_key):
            print(f'feature value: {feature_key} found, setting false')
            setattr(user, feature_key, False)
            db.session.commit()
            return jsonify({'success': True})
    else:
        print(f'feature value {feature_key} not found. user attribs: {user.__dict__}')
        app.logger.error(f'feature value {feature_key} not found. user attribs: {user.__dict__}')
        return jsonify({'success': False})

@login_required
@app.route('/')
def dashboard():
    if isinstance(current_user, User): # current_user can be anonymous and need to login
        try:
            user_id = current_user.id
            user = app_instance.user_datastore.find_user(id=user_id)
            if user:
                goals = Goal.query.filter_by(user_id=user_id).all()
                return render_template('dream-home.html', goals=goals)
        except Exception as e:
            app.logger.error(f'Dashboard loading error: {e}')
            return redirect(url_for('error_page', error_message='Error logging you in.'))
    return redirect(url_for('security.login'))
@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/chat_api/<string:goal_id>', methods=['POST'])
def chat_api(goal_id):
    print("a POST request was made to chat")
    # task = Task.query.filter_by(id=task_id).first()
    goal = Goal.query.filter_by(id=goal_id).first()
    message = request.json['message']
    chat_bot = GoalChatBot(goal)
    response = chat_bot.get_response(message)
    return jsonify({'response': response})

@app.route('/task_chat_api/<string:task_id>', methods=['POST'])
def task_chat_api(task_id):
    task = Task.query.filter_by(id=task_id).first()
    goal = Goal.query.filter_by(id=task.goal_id).first()
    chat_bot = TaskChatBot(task, goal, [task.get_task() for task in task.subtasks])
