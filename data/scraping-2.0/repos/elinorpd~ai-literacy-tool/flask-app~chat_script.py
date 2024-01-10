import openai
import os
import json
import inquirer
import datetime 
import argparse
import time

# Set up OpenAI API credentials
openai.api_key = os.environ["OPENAI_API_KEY"]

def editable_str(editable, value):
    """helper function to add editable string to lesson plan"""
    if editable:
        return " (editable: " + str(value) + "): " 
    else:
        return ": "


def parse_lesson_plan_html(lesson_plan):
    html_str = ""

    # Title
    if lesson_plan.get('title'):
        html_str += f'<h3>Title: {lesson_plan["title"]}</h3><br/><br/>\n'

    # Duration
    if lesson_plan.get('duration'):
        html_str += f'<h3>Duration: {lesson_plan["duration"]} mins.</h3><br/><br/>\n'

    # Overview
    if lesson_plan.get('overview'):
        html_str += f'<h3>Overview:</h3><br/><p>{lesson_plan["overview"]}</p><br/>\n'

    # Audience
    if lesson_plan.get('audience'):
        html_str += f'<h3>Audience:</h3><br/><p>{lesson_plan["audience"]}</p><br/>\n'

    # Learning Objectives
    if lesson_plan.get('objectives'):
        html_str += f'<h3>Learning Objectives:</h3><br/><p>{lesson_plan["objectives"]}</p><br/>\n'

    # AI Literacy Learning Objectives
    ailit_objectives = lesson_plan.get('ailitobjectives', [])
    custom_objective = lesson_plan.get('customobjective')
    if any(obj.get('checked') for obj in ailit_objectives) or custom_objective:
        html_str += f'<h3>AI Literacy Learning Objectives:</h3><br/><ul>\n'
        for obj in ailit_objectives:
            if obj.get('checked'):
                html_str += f'<li>{obj["label"]}</li>\n'
        if custom_objective:
            html_str += f'<li>{custom_objective}</li>\n'
        html_str += '</ul><br/>\n'
    activity = lesson_plan.get('activity', {})
    if activity.get('title'):
        html_str += f'<h3>Activity Title:</h3><br/><p>{activity.get("title", "")}</p><br/>\n'
        dur = activity.get("duration", "")
        html_str += f'<h3>AI Activity:</h3><br/><p>Duration {dur} minutes.</p><br/>\n'
        html_str += f'<h5>Activity Description:</h5><br/><p>{activity.get("description", "")}</p><br/>\n'
        if activity.get('assessment'):
            html_str += f'<h5>Activity Assessment:</h5><br/><p>Create an activity assessment consisting of 3-4 multiple choice quiz questions complete with all answer choices and an explanation of the correct answer choice. Please format it properly in a list.</p><br/>\n'
        if activity.get('alternatives'):
            html_str += f'<h5>Activity Alternatives:</h5><br/><p>Please suggest a version of this activity for lower and more advanced level students.{" If necessary, include versions of the assessment appropriate for the two levels." if activity.get("assessment") else ""}</p><br/>\n'

    # AI Activity
    ai_activity = lesson_plan.get('aiactivity', {})
    aiactivity = lesson_plan.get('aiactivity', {}).get('duration') # is true only if ai activity is there
    if aiactivity:
        dur = ai_activity.get("duration", "")
        html_str += f'<h3>AI Activity:</h3><br/><p>Duration {dur} minutes.</p><br/>\n'
        req = ai_activity.get('req', '') # needed to make this a var bc of too many nested quotes lol
        html_str += f'<h5>AI Activity Description:</h5><br/><p>Please design and describe an AI activity relevant to both the lesson and the AI Literacy Objectives above in as much detail as needed for a teacher with no AI experience to implement this in their classroom {f"for {dur} minutes" if dur else ""}.{f" Please adhere to the following requirements: {req}" if req else ""}</p><br/>\n'
        if ai_activity.get('assessment'):
            html_str += f'<h5>AI Activity Assessment:</h5><br/><p>Create an activity assessment consisting of 3-4 multiple choice quiz questions complete with all answer choices and an explanation of the correct answer choice. Please format it properly in a list.</p><br/>\n'
        if ai_activity.get('alternatives'):
            html_str += f'<h5>Activity Alternatives:</h5><br/><p>Please suggest a version of this activity for lower and more advanced level students.{" If necessary, include versions of the assessment appropriate for the two levels." if ai_activity.get("assessment") else ""}</p><br/>\n'

    return html_str, aiactivity

#function to load and parse json file containing the lesson plan
def parse_lesson_plan_cli(file_path, editable=True):
    f = open(file_path)
    data = json.load(f)
    lesson_plan = ""
    for key in data:
        if key == 'lesson_title':
            lesson_plan += "Title" + editable_str(editable, data[key]['editable']) + data[key]['value'] + "\n"
        elif key == 'duration':
            lesson_plan += "Duration" + editable_str(editable, data[key]['editable']) + str(data[key]['value']) + " mins\n"
        elif key == 'learning_objectives':
            lesson_plan += "Learning Objectives:\n"
            for id, obj in enumerate(data[key], start=1):
                lesson_plan += "\t" + str(id) + editable_str(editable, obj['editable']) + obj['value'] + "\n"
        elif key == 'discussions':
            lesson_plan += "Discussions:\n"
            for disc in data[key]:
                lesson_plan += "\t- Title" + editable_str(editable, disc['editable']) + disc['title'] + "\n"
                lesson_plan += "\tDescription" + editable_str(editable, disc['editable']) + disc['description'] + "\n"
        elif key == 'activities':
            lesson_plan += "Activities:\n"
            for act in data[key]:
                lesson_plan += "\t- Title" + editable_str(editable, act['editable']) + act['title'] + "\n"
                lesson_plan += "\tDescription" + editable_str(editable, act['editable']) + act['description'] + "\n"
            if 'ai_activity_duration' in data and data['ai_activity_duration']['value'] > 0:
                lesson_plan += "\t- AI Activity (editable: True):\n"
                lesson_plan += "\tDescription (editable: True): " + f"{str(data['ai_activity_duration']['value'])} minute activity goes here.\n"
        elif key == 'custom':
            # lesson_plan += "Additional Components:\n"
            for cust in data[key]:
                lesson_plan += cust['type'] + ":\n"
                lesson_plan += "Title" + editable_str(editable, cust['editable']) + cust['title'] + "\n"
                lesson_plan += "Description" + editable_str(editable, cust['editable']) + cust['description'] + "\n"            
    return lesson_plan
        
#function to load and print json file containing the lesson plan
def print_lesson_plan(file_name):
    f = open(file_name)
    data = json.load(f)
    # pretty print the json
    print(json.dumps(data, indent=4))
    
#function to read multiline input
def get_multiline_input():
    print("Double-enter to save it.")
    content = []
    while True:
        try:
            data = input()
            content.append(data)
            if not data: 
                content_str = "\n".join(content[:-1])
                return content_str
        except KeyboardInterrupt:
            return

# Define a function to generate a response from GPT-4
def generate_response(lessonplan, args=None, save=True, html=True, ai_activity=False):
    """
    Generates a new lesson plan by modifying editable sections of an existing lesson plan using OpenAI's GPT-3 language model.

    Args:
        lessonplan (str): lesson plan in dictionary/json format
        args (argparse.Namespace, optional): Command-line arguments. Defaults to None.

    Returns:
        str: The original lesson plan with modified editable sections, written to a new file.
    """   
    
    lesson_plan, aiactivity = parse_lesson_plan_html(lessonplan)
    print(lesson_plan)
    html_str = "html formatting within a <p></p>. Add a <br/> before every <b>. List items in newline" if html else "plain text formatting"
    activity_str = "incorporate a new activity based on given AI literacy learning objectives." if aiactivity else "modify an existing activity according to the instructions in the text."

    start_time = time.time()
    response = openai.chat.completions.create(
                  model=args.model if args else "gpt-3.5-turbo",
                  messages=[
                      {"role": "system", "content": f"You are an expert in {'AI literacy and' if aiactivity else ''} middle school education. We will give you an existing lesson plan from a teacher.\
                        Your task is to modify the lesson plan to {activity_str}. Replace the incomplete activity below with a topic, age, and level-appropriate activity according to the target audience (if not provided, assume middle school ages 11-14). Be specific and include all necessary details for a teacher to implement.\
                        For other sections (title, overview, lesson objectives), modify if you think it is necessary to maintain coherence {'(e.g. incorporate AI Literacy aspect to lesson overview)' if aiactivity else ''}. Do not edit other sections!\
                        \n\nReturn only the lesson plan in {html_str}, with your edits with NO ADDITIONAL TEXT OR REFERENCE TO YOUR EDITS."},
                      {"role": "user", "content": str(lesson_plan)},
                  ],
                  )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Time to generate response: {elapsed_time} seconds")
    
    result = response.choices[0].message.content
    print("Here is the new lesson plan:")
    print(result)
    if save: # save the file to output_dir
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
        if not args:
            output_dir = os.path.join(os.path.abspath(os.getcwd()), args.output_dir)
        else:
            output_dir = os.path.abspath(os.getcwd())
        # check if output_dir is a directory, if not, create it
        if os.path.isdir(output_dir) == False:
            os.mkdir(output_dir)
        file_name = 'ai_lesson_plan_' + now + '.txt'
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, 'w') as fp:
            fp.write(result)
    
    return result

#function to input and save lesson plan
def input_lesson_form(args):
    if args.input:
        lesson_plan = json.load(open(args.input))
    else:
        lesson_plan={}
    form = 1
    
    # TODO(eventually) dont allow empty inputs
    
    print("Hello! Please enter your current lesson plan. At a minimum, you must give a Title, Learning Objectives, and Duration.\n\
        For each component, you will be asked whether it is 'editable.' This means that the AI will potentially modify this section in\n\
            order to add AI Literacy learning objectives and/or activities. Ideally, you should include at least one activity that is editable.\n\
            If not, the AI will suggest a new AI literacy activity for you.")
    while form:
        form_choice = [
        inquirer.List('lesson_parameters',
                    message="What would you like to input? When you are done, select 'Save Form'",
                    choices=['Title', 'Learning Objectives', 'Duration', 'Discussions', 'Activities', 'Custom Component', 'Save Form'],
                ),
        ]

        lesson_parameter = inquirer.prompt(form_choice)['lesson_parameters']

        confirmation = True
        cur_dict = {}
        if lesson_parameter == "Title": 
            if 'lesson_title' in lesson_plan:
                print("Current Title:\n", lesson_plan['lesson_title']['value'], ", Editable: ",lesson_plan['lesson_title']['editable'])

                confirm = {
                    inquirer.Confirm('confirmation',
                        message="Do you want to enter a new title?" ,
                        default=False),
                }
                confirmation = inquirer.prompt(confirm)['confirmation']
                print(confirmation)
            if confirmation == True:
                lesson_details = {
                    inquirer.Text("parameter", message="Enter the title of the lesson"),
                    inquirer.Confirm('confirmation',
                        message="Do you want to make 'Title' an editable parameter by the AI?" ,
                        default=False),
                }
                inq_var = inquirer.prompt(lesson_details)
                cur_dict['value'] = inq_var['parameter']
                cur_dict['editable'] = inq_var['confirmation']
                lesson_plan['lesson_title'] = cur_dict


        if lesson_parameter == "Duration": 
            if 'duration' in lesson_plan:
                print("Current Duration:\n ", lesson_plan['duration']['value'], "mins, Editable: ", lesson_plan['duration']['editable'])
                confirm = {
                    inquirer.Confirm('confirmation',
                        message="Do you want to enter a new duration?" ,
                        default=False),
                }
                confirmation = inquirer.prompt(confirm)['confirmation']
            if confirmation == True:
                lesson_details = {
                    inquirer.Text("parameter", message="Enter the duration of the lesson in minutes"),
                    inquirer.Confirm('confirmation',
                        message="Do you want to make 'Duration' an editable parameter by the AI?" ,
                        default=False),
                }
                inq_var = inquirer.prompt(lesson_details)
                cur_dict['value'] = int(inq_var['parameter'])
                cur_dict['editable'] = inq_var['confirmation']
                lesson_plan['duration'] = cur_dict
   

        elif lesson_parameter == "Learning Objectives":
            # TODO reevaluate whether to keep as a list or make it one string with bullets/newlines. current upside is editable for each objective, but i honestly dont think its necessary, unless we always allow the AI to add new objectives
            if 'learning_objectives' in lesson_plan:
                print("Current Learning Objectives:\n")
                for item in lesson_plan['learning_objectives']:
                    print("-",item['value'])
                    print("Editable: ", item['editable'])
                confirm = {
                    inquirer.Confirm('confirmation',
                     message="Do you want to enter a new learning objective?" ,
                     default=False),
                }
                confirmation = inquirer.prompt(confirm)['confirmation']
            
            if confirmation == True:
                print("Enter learning objective for the lesson:")
                user_input= get_multiline_input()
                cur_dict['value'] = user_input

                lesson_details = {
                    # inquirer.Text("parameter", message="Enter learning objective for the lesson"),
                    inquirer.Confirm('confirmation',
                        message="Do you want to make this 'Learning Objective' an editable parameter by the AI?" ,
                        default=False),
                }
                inq_var = inquirer.prompt(lesson_details)
                # cur_dict['value'] = inq_var['parameter']
                cur_dict['editable'] = inq_var['confirmation']
                if 'learning_objectives' in lesson_plan:
                    lesson_plan['learning_objectives'].append(cur_dict)
                else:
                    lesson_plan['learning_objectives']=[]
                    lesson_plan['learning_objectives'].append(cur_dict)


        elif lesson_parameter == "Discussions":
            if 'discussions' in lesson_plan:
                print("Current Discussions:\n")
                for item in lesson_plan['discussions']:
                    print("-Title: ",item['title'])
                    print("Description: ",item['description'])
                    print("Editable: ", item['editable'])
                confirm = {
                    inquirer.Confirm('confirmation',
                     message="Do you want to enter a new discussion?" ,
                     default=False),
                }
                confirmation = inquirer.prompt(confirm)['confirmation']
            
            if confirmation == True:
                user_input = input("Enter title for the discussion:")
                cur_dict['title'] = user_input

                print("Enter description of discussion:")
                user_input= get_multiline_input()
                cur_dict['description'] = user_input

                lesson_details = {
                    # inquirer.Text("parameter", message="Enter discussion title for the lesson"),
                    # inquirer.Text("parameter_descr", message="Enter description of discussion"),
                    inquirer.Confirm('confirmation',
                        message="Do you want to make this 'Discussion' an editable parameter by the AI?" ,
                        default=False),
                }
                inq_var = inquirer.prompt(lesson_details)
                cur_dict['editable'] = inq_var['confirmation']
                if 'discussions' in lesson_plan:
                    lesson_plan['discussions'].append(cur_dict)
                else:
                    lesson_plan['discussions']=[]
                    lesson_plan['discussions'].append(cur_dict)

        elif lesson_parameter == "Activities":
            if 'activities' in lesson_plan:
                print("Current Activities:\n")
                for item in lesson_plan['activities']:
                    print("-Title: ",item['title'])
                    print("Description: ",item['description'])
                    print("Editable: ", item['editable'])
                confirm = {
                    inquirer.Confirm('confirmation',
                     message="Do you want to enter a new activity?" ,
                     default=False),
                }
                confirmation = inquirer.prompt(confirm)['confirmation']
            
            if confirmation == True:
                user_input = input("Enter activity title:")
                cur_dict['title'] = user_input

                print("Enter description of activity (including desired length in minutes, if applicable):")
                user_input= get_multiline_input()
                cur_dict['description'] = user_input

                lesson_details = {
                    # inquirer.Text("parameter", message="Enter activity title for the lesson"),
                    # inquirer.Text("parameter_descr", message="Enter description of activity"),
                    inquirer.Confirm('confirmation',
                        message="Do you want to make this 'Activity' an editable parameter by the AI?" ,
                        default=False),
                }
                inq_var = inquirer.prompt(lesson_details)
                cur_dict['editable'] = inq_var['confirmation']
                if 'activities' in lesson_plan:
                    lesson_plan['activities'].append(cur_dict)
                else:
                    lesson_plan['activities']=[]
                    lesson_plan['activities'].append(cur_dict)
                    
        elif lesson_parameter == "Custom Component":
            if confirmation == True:
                user_input = input("Enter type of component:")
                cur_dict['type'] = user_input

                user_input = input("Enter a title:")
                cur_dict['title'] = user_input

                print("Enter a description:")
                user_input= get_multiline_input()
                cur_dict['description'] = user_input

                lesson_details = {
                    inquirer.Confirm('confirmation',
                        message="Do you want to make this 'Custom Component' an editable parameter by the AI?" ,
                        default=False),
                }
                inq_var = inquirer.prompt(lesson_details)
                cur_dict['editable'] = inq_var['confirmation']
                if 'custom' in lesson_plan:
                    lesson_plan['custom'].append(cur_dict)
                else:
                    lesson_plan['custom']=[]
                    lesson_plan['custom'].append(cur_dict)
        
        elif lesson_parameter == "Save Form":
            edit_activities = False
            if 'activities' in lesson_plan:
                for act in lesson_plan['activities']:
                    # check if any activities are editable, if none are, 
                    # then ask if they want a new activity with ai lit incorporated and length as below
                    if act['editable']: # == True:
                        edit_activities = True
                        break
            if edit_activities == False:
                print("None of your activities are marked as editable for the AI. Would you like the AI to suggest a new AI literacy activity?")
                lesson_details = {
                    inquirer.Text("parameter", message="If so, please enter the activity length in minutes. If not, enter 0")
                    }
                inq_var = inquirer.prompt(lesson_details)
                cur_dict['value'] = int(inq_var['parameter'])
                cur_dict['editable'] = True
                lesson_plan['ai_activity_duration'] = cur_dict
                lesson_plan['duration']['value'] += lesson_plan['ai_activity_duration']['value']
            
            print("Here is the saved lesson plan:")
            print(json.dumps(lesson_plan, indent=4))
            now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
            output_dir = os.path.join(os.path.abspath(os.getcwd()), args.output_dir) # TODO use argparse to make this flexible
            if os.path.isdir(output_dir) == False:
                os.mkdir(output_dir)
            file_name = 'lesson_plan_' + now + '.json'
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, 'w') as fp:
                json.dump(lesson_plan, fp, indent=4)
            form = 0

    return file_path

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--input', type=str, default=None, help='optional input json file name')
    argparse.add_argument('--output_dir', type=str, default='output', help='optional output directory name (relative path), default "output"')
    argparse.add_argument('--model', type=str, default='gpt-3.5-turbo', help='openai model to use, default "gpt-3.5-turbo" or use "gpt-4"')
    args = argparse.parse_args()
    
    file_path = input_lesson_form(args)
    # print_lesson_plan(file_path)
    print(parse_lesson_plan_cli(file_path))
    # generate AI modified lesson plan
    print("Generating AI modified lesson plan...")
    generate_response(file_path, args)
