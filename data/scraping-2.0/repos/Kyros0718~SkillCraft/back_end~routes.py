# BackEnd/routes.py
from flask import Blueprint, render_template, request, jsonify
import openai, os

bp = Blueprint('main', __name__)
openai.api_key = os.environ.get("API_KEY")

# PAGE RENDERING
@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/instruction.html')
def instruction():
    return render_template('instruction.html')




#FUNCTIONALITY
@bp.route('/project_searching', methods=['POST'])
def search_project_process():
    
    data = request.get_json() #PULL JS file from client side

    input_text = data.get('input_text') #PULL input_text from JS
    skill_level = data.get('skill_level') #PULL skill_level from JS
    project_list = [] #A list of Projects

    input_response = openai.Completion.create( #AI Response to a prompt
        engine="text-davinci-003",
        prompt = f"Generate 4 project ideas that involve {input_text} at a {skill_level} level",
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.5,
    )

    raw_response = input_response['choices'][0]['text'].split('\n') #AI raw response split up
    main_response = list(filter(lambda x: x!='', raw_response)) #response without unnecessary spaces
    cut_response = list(map(lambda x: x[x.find(" ")+1:], main_response)) #non-numbered response

    project_list = cut_response

    project_info = {
        "input_text": input_text,
        "skill_level":skill_level,
        "project_list":project_list} #Data for JS
    return jsonify(project_info)


@bp.route('/walkthrough', methods=['POST'])
def walkthrough_process():
    
    data = request.get_json() #PULL JS file from client side

    input_text = data.get('input_text') #PULL input_text from JS
    skill_level = data.get('skill_level') #PULL skill_level from JS
    project_idea = data.get('project_idea') #PULL project_idea from JS
    project_idea_steps = [] #steps to project

    project_response = openai.Completion.create( #AI Response to a prompt
        engine="text-davinci-003",
        prompt = f"A user is trying to develop the project: '{project_idea}'. The user is trying to use '{input_text}' for this project and their skill level is '{skill_level}'. Generate steps to develop the project",
        max_tokens=3000,
        n=1,
        stop=None,
        temperature=0.5,
    )

    raw_response = project_response['choices'][0]['text'].split('\n') #AI raw response split up
    main_response = list(filter(lambda x: x!='', raw_response)) #response without unnecessary spaces
    modified_response = list(map(lambda x: "STEP "+x, main_response)) #non-numbered response
    if "STEP 1" not in modified_response[0]:
        project_idea_steps = modified_response[1:]
    else:
        project_idea_steps = modified_response

    project_info = {"input_text": input_text,
                    "skill_level":skill_level,
                    "project_idea":project_idea,
                    "project_idea_steps":project_idea_steps} #Data for JS
    return jsonify(project_info)


@bp.route('/expanding-instruction', methods=['POST'])
def expanding_instruction():
    data = request.get_json() #PULL JS file from client side

    input_text = data.get('input_text') #PULL input_text from JS
    skill_level = data.get('skill_level') #PULL skill_level from JS
    project_idea = data.get('project_idea') #PULL project_idea from JS
    project_idea_steps = data.get('project_idea_steps')#steps to project
    project_idea_currentstep = data.get('project_idea_currentstep')
    project_step_help = data.get('project_step_help')
    project_step_expand = []
    walkthrough_response = openai.Completion.create( #AI Response to a prompt
        engine="text-davinci-003",
        prompt = f"""A user is trying to develop the project: '{project_idea}'. 
            The user is trying to use '{input_text}' for this project and their skill level is '{skill_level}'. 
            The User is shown this guide: {project_idea_steps}. The User is on '{project_idea_currentstep}'. 
            However, the User is confused on the idea of {project_step_help}. 
            Generate a walkthrough for this idea""",
        max_tokens=3000,
        n=1,
        stop=None,
        temperature=0.5,
    )
    raw_response = walkthrough_response['choices'][0]['text'].split('\n')
    main_response = list(filter(lambda x: x!='', raw_response)) #response without unnecessary spaces
    project_step_expand = main_response

    project_info = {"input_text": input_text,
                    "skill_level": skill_level,
                    "project_idea": project_idea,
                    "project_idea_steps": project_idea_steps,
                    "project_idea_currentstep": project_idea_currentstep,
                    "project_step_expand": project_step_expand} #Data for JS
    return jsonify(project_info)