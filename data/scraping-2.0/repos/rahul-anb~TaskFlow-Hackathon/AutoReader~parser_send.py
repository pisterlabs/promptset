from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain.llms import OpenAI
from dotenv import load_dotenv

load_dotenv()

import json

def generate_task():
    llm = OpenAI(temperature=0.6)
    
    prompt = """you are a Task list creation aiding tool who will take in a input detail and give out an output in a specific format to aid in generating tasks and subtask for a project.
    Generated tasks should be very project specific and include specialized tasks in them.
    Create only 4 to 6 main headings unless specifically asked to.
    Create a task list for inventory management system. Let numbering for main headings be numbers and numbering for subheadings be alphabets."""
    name = llm(prompt)
    
    return name

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'



@app.route('/convert-json', methods=['POST'])
def call():
    data= generate_task()
    # data = open('data.txt', 'r').read()
    data = data[2:]
        

    data = data.split('\n')
    title = data[0]
    data = data[2:]
    # print(title)

    #strip elementes of list
    data = [i.strip() for i in data]


    line = []
    start = 0
    # print(data)

    start = 0
    for i in range(1,len(data)):
        if data[i] == '':
            line.append(data[start:i])
            start = i+1
        
    for i in line:
        print(i)
        
        
    template_input = {
                        "id": "", 
                        "type": "input", 
                        "data": { "label": "" }, 
                        "position": {"x": 0, "y": 0}, 
                        "sourcePosition":"right"
                    }

    template_selectorNode = {
                        "id": "", 
                        "type": "selectorNode", 
                        "data": { "title": "", "deadline":"", "options":[], "personAssigned":"" }, 
                        "position": {"x": 0, "y": 0}
                    }
        
    template_output = {
                        "id": "", 
                        "type": "output", 
                        "data": { "label": "" }, 
                        "position": {"x": 0, "y": 0}, 
                        "targetPosition":"left"
                    }

            
            
    setNodes = []
    startx, starty = 500, 500

    screen_height = 1080/len(line)
    template_input = {
                        "id": 0, 
                        "type": "input", 
                        "data": { "label": title }, 
                        "position": {"x": startx, "y": starty}, 
                        "sourcePosition":"right"
                    }

    setNodes.append(template_input)

    for i in range(len(line)):
        template_selectorNode = {
                        "id": i+1, 
                        "type": "selectorNode", 
                        "data": { "title": line[i][1], "deadline":"", "options":line[i][1:], "personAssigned":"" }, 
                        "position": {"x": startx+200, "y": starty+screen_height*(i+1)}
                    }
        setNodes.append(template_selectorNode)

    setEdges = []
    template_edges = {
                        "id": "", 
                        "source": "", 
                        "target": "",
                        "sourceHandle": 'a',
                        "animated": "true",
                        "style": { "stroke": '#fff' }
                    }


    for i in range(1,len(setNodes)):
        template_edges = {
                        "id": 30+i, 
                        "source": 0, 
                        "target": i,
                        "animated": "true",
                        "style": { "stroke": '#fff' }
                    }
        setEdges.append(template_edges)

    for i in range(31,len(setNodes)+1):
        template_edges = {
                        "id": 50+i, 
                        "source": i, 
                        "target": i+1,
                        "animated": "true",
                        "style": { "stroke": '#fff' }
                    }
        setEdges.append(template_edges)
        
        
    
    json_setNodes = json.dumps(setNodes, indent=4)
    json_setEdges = json.dumps(setEdges, indent=4)
    jsonify_data = {"setNodes":setNodes, "setEdges":setEdges}
    print(json_setNodes, json_setEdges)
    # return jsonify(jsonify_data), 200
    
    
    
if __name__ == '__main__':
    # app.run(debug=True)
    call()
    