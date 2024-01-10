from flask import Flask, send_from_directory, redirect, request, render_template
from base64 import b64decode
import os
import cv2
import numpy as np
import openai
import json
from pydantic import BaseModel
import time
import shutil

def byte_image_to_numpy(byte_image):
    np_array = np.frombuffer(byte_image, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return img

def show(img, wait=0):
    if type(img) is bytes:
        img = byte_image_to_numpy(img)
    cv2.imshow("img", img)
    cv2.waitKey(wait)

with open("../config.txt", "r") as f:
    api_key = f.read().strip()
    openai.api_key = api_key
    f.close()

def generateImage(prompt, debug=False):
    if debug:
        print("Debug mode is on, skipping image generation")
        return open("test.png", "rb").read()
    
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="256x256",
        response_format="b64_json",
    )

    image_data = b64decode(response["data"][0]["b64_json"])
    return image_data

class WebsiteAIResponse(BaseModel):
    html: str
    image_names: list[str]
    image_prompts: list[str]

def generate(userRequest, model="gpt-3.5-turbo-0613", messages=None):
    msg = [
            {"role": "system", "content": "You are a machine that generates a website with HTML."},
            {"role": "user", "content": f"The request is: {userRequest}. Create HTML code with all of the content in English that would be in the request. The output must be valid json text without any unescaped double quotes and no newline characters. Be informative. Use between one and three images. In the backend, make corresponding lists of image file names and of detailed descriptions for each image name. Position these images in the website where they logically make sense. The folder for images is called \"images\". Use bootstrap CSS classes to style the html. Do not link a bootstrap stylesheet."}
        ]
    
    if messages is not None:
        msg.extend(messages)

    response = openai.ChatCompletion.create(
        model=model,
        messages=msg,
        functions=[
            {
            "name": "create_website",
            "description": "Create a website based on the given request and create image prompts for the images in the website",
            "parameters": WebsiteAIResponse.model_json_schema()
            }
        ],
        function_call={"name": "create_website"}
    )
    with open("json.json", "w") as f:
        f.write(response.choices[0]["message"]["function_call"]["arguments"])
        f.close()
    try:
        output = json.loads(response.choices[0]["message"]["function_call"]["arguments"].strip().encode())
    except json.decoder.JSONDecodeError:
        return "failed", [], []
    return output["html"], output["image_names"], output["image_prompts"]

app = Flask(__name__, static_folder="static")

@app.route('/', methods=['GET'])
def index():
    return redirect('/home')

global lastQuery, app_root, generations_count, view_number, project_number, project_path, projects_count
lastQuery = ""

app_root = os.path.dirname(os.path.abspath(__file__))
projects_count = len([entry for entry in os.listdir(app_root) if os.path.isdir(os.path.join(app_root, entry)) and entry.startswith("generations")])

project_number = projects_count - 1 if projects_count > 0 else 0
project_path = os.path.join(app_root, f"generations{project_number}")

generations_count = 0
if os.path.exists(project_path):
    generations_count = len([entry for entry in os.listdir(project_path) if os.path.isfile(os.path.join(project_path, entry)) and entry.startswith("baseHTML")])

view_number = generations_count - 1 if generations_count > 0 else 0

def processHTML(html, current_view=view_number):
    if current_view == 0: prevView = 0
    else: prevView = current_view - 1
    if current_view == generations_count - 1: nextView = current_view
    else: nextView = current_view + 1
    
    insertIdx = html.find("<head>")
    element = "\n<link rel='stylesheet' href='css/{{stylesheet}}'>"
    html = html[:insertIdx+6] + element + html[insertIdx+6:]

    html = html[:insertIdx+6+len(element)] + "\n<link rel='stylesheet' href='css/genstyle.css'>\n<script src='https://ajax.googleapis.com/ajax/libs/jquery/3.7.1/jquery.min.js'></script>\n<script src='https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js'></script>\n" + html[insertIdx+6+len(element):]
    
    insertIdx = html.find("<body>")
    html = html[:insertIdx+6] + f"\n<div class='floating-menu'>\n<div id='stylesheetlistcontainer'><p>Stylesheets:</p><div id='stylesheetlist'>\n<a class='button' href='/lastgen?sheet=cerulean.css&project={'{{project_num}}'}&view={'{{view}}'}'>Cerulean</a>\n<a class='button' href='/lastgen?sheet=cosmo.css&project={'{{project_num}}'}&view={'{{view}}'}'>Cosmo</a>\n<a class='button' href='/lastgen?sheet=darkly.css&project={'{{project_num}}'}&view={'{{view}}'}'>Darkly</a>\n<a class='button' href='/lastgen?sheet=journal.css&project={'{{project_num}}'}&view={'{{view}}'}'>Journal</a>\n<a class='button' href='/lastgen?sheet=lux.css&project={'{{project_num}}'}&view={'{{view}}'}'>Lux</a>\n<a class='button' href='/lastgen?sheet=quartz.css&project={'{{project_num}}'}&view={'{{view}}'}'>Quartz</a>\n<a class='button' href='/lastgen?sheet=united.css&project={'{{project_num}}'}&view={'{{view}}'}'>United</a></div>\n</div>\n<form action='/lastgen' onsubmit='hidesubmit();'>\n<p>Last feedback: \"{'{{view_feedback}}'}\"</p>\n<textarea rows='3' name='feedback' placeholder='enter feedback'></textarea>\n<div class='submitbutton'>\n<button id='submitbutton' type='submit'>Submit</button><div class='loader' id='hiddenDiv'>\n</div>\n</div>\n<input type='hidden' name='sheet' value='{'{{stylesheet}}'}'>\n<input type='hidden' name='view' value='{'{{view}}'}'>\n<input type='hidden' name='project' value='{'{{project_num}}'}'>\n</form>\n<div class='menu-section'>\n<div id='view-nav'>\n<a class='button' href='/lastgen?sheet={'{{stylesheet}}'}&view={prevView}&project={'{{project_num}}'}'>&lt;</a>\n<div id='generation-view'>\n<p>Revision: #{current_view} (made from Revision #{'{{source_num}}'})</p>\n<p>Total Generations: {generations_count}</p>\n</div>\n<a class='button' href='/lastgen?sheet={'{{stylesheet}}'}&view={nextView}&project={'{{project_num}}'}'>&gt;</a>\n</div>\n<a class='button home' href='/home'>Home</a>\n</div>\n</div>\n<script>\nlet showButton = document.getElementById('submitbutton');\nlet hiddenDiv = document.getElementById('hiddenDiv');\nshowButton.onclick = function() {{hiddenDiv.style.display = 'block';}};\nfunction hidesubmit() {{let button = document.getElementById('submitbutton');button.disabled = true; }}</script>" + html[insertIdx+6:]
    return html

@app.route('/home', methods=['GET'])
def home():
    global lastQuery, app_root, generations_count, view_number, project_number, project_path, projects_count
    if request.method == 'GET' and "request" in request.args:
        if request.args["request"] == "": return redirect('/home')
        elif request.args["request"] == lastQuery: 
            stylesheet = "journal.css"
            if request.method == 'GET' and "sheet" in request.args: stylesheet=request.args["sheet"]
            return redirect(f"/lastgen?sheet={stylesheet}")
        else: 
            print("Request: " + request.args["request"])
            print("Image Generation:", "off" if "imagegen" not in request.args else "on")
            startTime = time.time()
            
            project_number = projects_count
            project_path = os.path.join(app_root, f"generations{project_number}")
            if os.path.exists(project_path):
                confirm = input("Project already exists.  Do you want to continue? (y/n) ")
                if confirm.lower() == "n": return redirect('/home')
            os.makedirs(os.path.join(project_path, "images"))
            projects_count += 1

            print("generating website")
            userRequest = request.args["request"]
            
            startTextTime = time.time()
            
            model = "gpt-3.5-turbo-0613"
            # model = "gpt-4-1106-preview"
            html, image_names, image_prompts = generate(userRequest, model=model)
            textTimeElapsed = time.time() - startTextTime
            print("text generation time: " + str(textTimeElapsed))

            if html == "failed":
                print("Failed to generate HTML due to JSON error")
                try:
                    shutil.rmtree(project_path, ignore_errors=True)
                except Exception as e:
                    print(f'Failed to delete directory: {e}')
                projects_count -= 1
                return redirect('/error')

            generations_count = 0
            view_number = 0
            with open(os.path.join(project_path, f"baseHTML{generations_count}.html"), "w") as f:
                f.write(html)
                f.close()
        
            generations_count += 1
            html = processHTML(html)

            # Writes the generated site to the generated folder
            with open("templates/view.html", "wb") as f:
                f.write(html.encode())
                f.close()
            
            # Generates images for each image prompt
            print("generating images")
            imageStartTime = time.time()
            
            debug = "imagegen" not in request.args
            for name, prompt in zip(image_names, image_prompts):
                img = generateImage(prompt, debug=debug)
                with open(os.path.join(project_path, "images", name), "wb") as f:
                    f.write(img)
                    f.close()
            
            imageTimeElapsed = time.time() - imageStartTime
            print("image generation time: " + str(imageTimeElapsed))

            # Save the current query as the last query
            lastQuery = request.args["request"]
            print("serving generated site")
            
            totalTimeElapsed = time.time() - startTime
            print(totalTimeElapsed)
            
            with open("time.txt", "a") as f:
                f.write(f"{totalTimeElapsed},{textTimeElapsed},{imageTimeElapsed},{userRequest},{len(html)},model:{model},imagegen:{not debug}\n")
                f.close()
            
            with open(os.path.join(project_path, "log.json"), "w") as f:
                json.dump({str(view_number): [lastQuery, 0]}, f)
                f.close()

            stylesheet = "journal.css"
            if request.method == 'GET' and "sheet" in request.args: stylesheet=request.args["sheet"]
            return redirect(f"/lastgen?sheet={stylesheet}&view={view_number}")
    
    projects = []
    projects_count = len([entry for entry in os.listdir(app_root) if os.path.isdir(os.path.join(app_root, entry)) and entry.startswith("generations")])
    for i in range(projects_count):
        path = os.path.join(app_root, f"generations{i}")
        if os.path.exists(path):
            log = os.path.join(path, "log.json")
            if os.path.exists(log):
                with open(log, "r") as f:
                    project = json.load(f)
                    f.close()
                projects.append(project["0"][0])
            else:
                projects.append("Not Found.")
    
    return render_template("index.html", projects=projects)

@app.route('/error')
def error():
    return send_from_directory(app.static_folder, "error.html")

@app.route('/generated/<path:filename>')
def web_gen_assets(filename):
    return send_from_directory("static/generated", filename)

@app.route('/css/<path:filename>')
def css_assets(filename):
    return send_from_directory("static/css", filename)

@app.route('/images/<path:filename>')
def img_gen_assets(filename):
    return send_from_directory(os.path.join(project_path, "images"), filename)

@app.route('/lastgen', methods=['GET'])
def lastgen():
    global lastQuery, app_root, generations_count, view_number, project_number, project_path, projects_count
    stylesheet = "journal.css"
    if request.method == 'GET':
        if "sheet" in request.args: stylesheet=request.args["sheet"]
        if "project" in request.args: project_number = int(request.args["project"])
        else: project_number = projects_count - 1
        project_path = os.path.join(app_root, f"generations{project_number}")
        generations_count = 0
        if os.path.exists(project_path):
            generations_count = len([entry for entry in os.listdir(project_path) if os.path.isfile(os.path.join(project_path, entry)) and entry.startswith("baseHTML")])
        view_number = generations_count - 1 if generations_count > 0 else 0

        if "view" in request.args: view = int(request.args["view"])
        else: view = view_number
        if "feedback" in request.args:
            if request.args["feedback"] == "": return redirect(f"/lastgen?sheet={stylesheet}&view={view}&project={project_number}")
            try:
                with open(os.path.join(project_path, f"baseHTML{view}.html"), "r") as f:
                    html = f.read()
                    f.close()
            except FileNotFoundError:
                with open("templates/view.html", "r") as f:
                    html = f.read()
                    f.close()

            newMsgs = [{"role": "assistant", "content": html},
                       {'role': 'user', 'content': "The following feedback are changes that need to made to the code.  Add, remove, and change the code as needed.  The output should be valid json text.  Use bootstrap CSS when needed.  The feedback is: " + request.args["feedback"]}
                    ]
            print("processing feedback:", request.args["feedback"] if "feedback" in request.args else "")
            startTime = time.time()
            newhtml, image_names, image_prompts = generate(lastQuery, messages=newMsgs)
            totalTimeElapsed = time.time() - startTime
            
            if newhtml == "failed":
                print("Failed to generate HTML due to JSON error")
                return redirect('/error')
            
            view_number += 1
            generations_count += 1
            with open(os.path.join(project_path, f"baseHTML{view_number}.html"), "w") as f:
                f.write(newhtml)
                f.close()
            
            outhtml = processHTML(newhtml)
            with open("templates/view.html", "wb") as f:
                f.write(outhtml.encode())
                f.close()

            with open(os.path.join(project_path, "log.json"), "r") as f:
                view_feedback = json.load(f)
                f.close()
            view_feedback[str(view_number)] = [request.args["feedback"], view]
            with open(os.path.join(project_path, "log.json"), "w") as f:
                json.dump(view_feedback, f)
                f.close()

            with open("feedbackTimes.txt", "a") as f:
                f.write(f"{totalTimeElapsed},{request.args['feedback']},{len(newhtml) - len(html)}\n")
                f.close()
            
            return redirect(f"/lastgen?sheet={stylesheet}&view={view_number}&project={project_number}")
        else:
            with open(os.path.join(project_path, f"baseHTML{view}.html"), "r") as f:
                html = f.read()
                f.close()
            html = processHTML(html, current_view=view)
            with open("templates/view.html", "wb") as f:
                f.write(html.encode())
                f.close()

    view_feedback = "Not Found."
    source_num = "Not Found."
    log_path = os.path.join(project_path, "log.json")
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            log = json.load(f)
            if str(view) in log:
                view_feedback, source_num = log[str(view)]
            f.close()
    return render_template("view.html", stylesheet=stylesheet, view_feedback=view_feedback, view=view, source_num=source_num, project_num=project_number)

if __name__ == "__main__":
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(host="0.0.0.0", port=80)