"""Routes for dashboard"""
from datetime import datetime
import re
import boto3
from sqlalchemy import func
from flask import Blueprint, current_app, render_template, send_file, redirect, url_for, request
from flask_login import login_required, current_user
from iqsheets_app import db
from iqsheets_app.models import Prompt, Template
from iqsheets_app.utils.decorators import check_confirmed_mail
from iqsheets_app.openai import openai_chat
from .forms import FormelForm, SkriptForm, SqlForm, RegExForm

################
#### config ####
################

dashboard_blueprint = Blueprint('dashboard', __name__)

# initialize S3 client using boto3
s3_client = boto3.client(
    's3',
    aws_access_key_id=current_app.config['AWS_ACCESS_KEY'],
    aws_secret_access_key=current_app.config['AWS_SECRET_ACCESS_KEY'],
    region_name=current_app.config['AWS_REGION']
    )

# Mapping of prompt types to their respective forms
FORM_MAP = {
    "formula": FormelForm,
    "skripte": SkriptForm,
    "sql": SqlForm,
    "regex": RegExForm,
}

CLEAN_MAP = {
    "Excel - VBA": "vba",
    "GSheets - Apps": "javascript",
}

################
#### helpers ####
################

def find_function(text, prompt_type):
    """
    Function collect the formula of openai response only.

    Parameters:
    text (str): The text from which the pattern will be removed.
    prompt_type (str): The type of prompt to be removed from the text.

    Returns:
    str: The text with the specified pattern removed.
    """
    pattern = r"```" + prompt_type + "(.*?)```"
    # Using re.findall to find all occurrences of the pattern
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

def remove_pattern_from_text(text, prompt_type):
    """
    Function to remove a specific pattern from a given text string.

    Parameters:
    text (str): The text from which the pattern will be removed.
    prompt_type (str): The type of prompt to be removed from the text.

    Returns:
    str: The text with the specified pattern removed.
    """
    start_pattern = re.escape(r"```") + prompt_type
    end_pattern = re.escape(r"```")
    text = re.sub(start_pattern, '', text, flags=re.DOTALL)
    text = re.sub(end_pattern, '', text, flags=re.DOTALL)
    return text.strip()

def process_form_data(form_data, prompt_type):
    '''Function to handle prompt input form data from inserted form.
    Parameters:
        form data: The text from which the pattern will be removed.
    Returns: 
        Input for Prompt DB Model
    '''
    form_data['prompt_type'] = prompt_type
    keys = ["prompt_type", "excel_google", "vba_app", "formula_explain", "prompt"]
    input_prompt = []
    for key in keys:
        if key in form_data:
            input_prompt.append(form_data[key])
    # Form user info to prompt for OpenAI
    input_prompt = " ".join(input_prompt)
    result = openai_chat(input_prompt)
    answer = result.choices[0].message.content
    
    category, prompt = form_data["formula_explain"], form_data["prompt"]
    
    # Increasing the amount of prompts and total tokens when prompt is generated
    current_user.num_prompts += 1
    current_user.num_tokens += result.usage.total_tokens
    
    return prompt_type, category, prompt, answer

def prompt_output_handler(prompt_result, prompt_type, form_data):
    """
    Function to handle the OpenAi response for user.

    Parameters:
    text (str): The text from which the pattern will be removed.
    prompt_type (str): The type of prompt to be removed from the text.

    Returns:
    str: The text with the specified pattern removed.
    """
    # Extracting the part of the string from "sql" to "19"7
    print(form_data)
    if "vba_app" in form_data:
        print(CLEAN_MAP[form_data["vba_app"]].lower())
        formulas = find_function(prompt_result, CLEAN_MAP[form_data["vba_app"]].lower())
        reduced_answer = remove_pattern_from_text(prompt_result, CLEAN_MAP[form_data["vba_app"]].lower())
        print(prompt_result, prompt_type.lower(),formulas, reduced_answer)
    else:
        formulas = find_function(prompt_result, prompt_type.lower())
        reduced_answer = remove_pattern_from_text(prompt_result, prompt_type.lower())
        print(prompt_result, prompt_type.lower(),formulas, reduced_answer)
    
    return formulas, reduced_answer
################
#### routes ####
################

@dashboard_blueprint.route('/dashboard', methods=['GET'])
@login_required
@check_confirmed_mail
def dashboard():
    """User Dashboard page"""
    num_prompts = Prompt.query.filter_by(user_id=current_user.id).count()
    favorite_prompt =  Prompt.query.filter_by(user_id=current_user.id, favorite=True).count()
    time_saved = num_prompts * 0.5
    fav_prompt_type = db.session.query(Prompt.prompt_type,func.count(Prompt.id)).filter(
                        Prompt.user_id == current_user.id).group_by(Prompt.prompt_type).all()
    if fav_prompt_type:
        most_used = max(fav_prompt_type, key=lambda item: item[1])
        most_used = most_used[0].capitalize()
    else:
        most_used = "/"
    return render_template('dashboard/dashboard.html', num_prompts=num_prompts, 
                           favorites=favorite_prompt, most_used=most_used, 
                           time_saved=time_saved)

@dashboard_blueprint.route('/<prompt_type>', methods=['GET', 'POST'])
@login_required
@check_confirmed_mail
def prompter(prompt_type):
    """User Dashboard page"""
    if prompt_type not in FORM_MAP:
        # Redirect to the default dashboard page for invalid prompt types
        return redirect(url_for('dashboard.dashboard'))

    form = FORM_MAP[prompt_type]()
    return render_template(f"dashboard/{prompt_type}_page.html", form=form)

@dashboard_blueprint.route('/<prompt_type>/result', methods=['GET', 'POST'])
@login_required
@check_confirmed_mail
def formel(prompt_type):
    """User Dashboard page"""
    if prompt_type not in FORM_MAP:
        # Redirect to the default dashboard page for invalid prompt types
        return redirect(url_for('dashboard.dashboard'))

    form = FORM_MAP[prompt_type]() 
   
    if request.method == 'POST' and form.validate_on_submit():
        form_data = form.data
        prompt_type, category, prompt, answer = process_form_data(form_data, prompt_type)
        
        # Creating prompt instance
        prompt = Prompt(user_id = current_user.id, prompt_type=prompt_type, 
                        category=category, prompt=prompt, result=answer)
        # Commiting prompt and numbers to db
        db.session.add(prompt)
        db.session.commit()
        
    if prompt.category == 'Erstellen' and prompt.prompt_type != "formula":
        formulas, reduced_answer = prompt_output_handler(prompt.result, prompt.prompt_type, form.data)
        return render_template(f'dashboard/{prompt_type}_page.html', answer=reduced_answer, form=form, prompt_id=prompt.id, formulas=formulas)    
    else:
        return render_template(f'dashboard/{prompt_type}_page.html', answer=prompt.result, form=form, prompt_id=prompt.id)
        
    return render_template(f'dashboard/{prompt_type}_page.html', form=form)

@dashboard_blueprint.route('/dashboard/favorite/<int:prompt_id>', methods=['GET'])
@login_required
@check_confirmed_mail
def prompt_favorite(prompt_id):
    ''' handles user feedback per prompt '''
    prompt = Prompt.query.filter_by(id=prompt_id).first()
    prompt.favorite = True
    db.session.commit()
    return redirect(url_for('dashboard.favorites'))

@dashboard_blueprint.route('/dashboard/negative/<int:prompt_id>', methods=['GET'])
@login_required
@check_confirmed_mail
def negative_feedback(prompt_id):
    ''' handles user feedback per prompt '''
    prompt = Prompt.query.filter_by(id=prompt_id).first()
    prompt.feedback = False
    db.session.add(prompt)
    db.session.commit()
    return redirect(request.referrer or '/default-page')

@dashboard_blueprint.route('/favoriten', methods=['GET', 'POST'])
@login_required
@check_confirmed_mail
def favorites():
    """User favorite Excel Formulas"""
    page = request.args.get('page', 1, type=int)
    favorite_formulas = Prompt.query.filter_by(user_id=current_user.id, 
                                               favorite=True).order_by(Prompt.created_at).paginate(page=page, 
                                                                                                   per_page=30)
    # prompt_types = db.session.query(Prompt.prompt_type).distinct().all()
    prompt_types = ["formula", "skripte", "sql", "regex"]
    today = datetime.now()
    if request.method == 'POST' and request.form['filter_value'] == "Alle":
    
        page = request.args.get('page', 1, type=int)
        favorite_formulas = Prompt.query.filter_by(user_id=current_user.id, favorite=True).order_by(Prompt.created_at).paginate(page=page, per_page=9)
    
    elif request.method == 'POST':
        filter_value = request.form['filter_value']
        page = request.args.get('page', 1, type=int)
        favorite_formulas = Prompt.query.filter_by(user_id=current_user.id, favorite=True, 
                                                   prompt_type=filter_value).order_by(Prompt.created_at).paginate(page=page, per_page=30)
        
        
    return render_template('dashboard/favorites.html', favorite_formulas=favorite_formulas, prompt_types=prompt_types, today=today)

@dashboard_blueprint.route('/formel_<int:favorite_id>/delete', methods=['GET'])
@login_required
@check_confirmed_mail
def delete_favorite(favorite_id):
    """Delete Formula/VBA to User favorites"""
    favorite = Prompt.query.filter_by(id=favorite_id).first()
    db.session.delete(favorite)
    db.session.commit()
    return redirect(url_for('dashboard.favorites'))
    
@dashboard_blueprint.route('/templates', methods=['GET', 'POST'])
@login_required
@check_confirmed_mail
def templates():
    """ Route for templates """
    page = request.args.get('page', 1, type=int)
    templates = Template.query.order_by(Template.created_at).paginate(page=page, per_page=12)
    categorys = db.session.query(Template.template_category).distinct().all()
        
    if request.method == 'POST' and request.form['filter_value'] == "Alle":
        page = request.args.get('page', 1, type=int)
        templates = Template.query.order_by(Template.created_at).paginate(page=page, per_page=12)
        
    elif request.method == 'POST':
        filter_value = request.form['filter_value']
        page = request.args.get('page', 1, type=int)
        templates = Template.query.filter_by(template_category=filter_value).order_by(Template.created_at).paginate(page=page, per_page=12)
            
    return render_template('dashboard/templates.html', templates=templates, categorys=categorys)

@dashboard_blueprint.route('/download', methods=['GET'])
@login_required
@check_confirmed_mail
def download():
    """ Route for templates download """
    filename = 'static/xlxs_templates/Calendar-Template.xlsx'
    try:
        return send_file(filename)
    except Exception as e:
        return str(e)
