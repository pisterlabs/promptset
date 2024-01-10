import os
import tempfile
from flask import Flask, request, send_file, send_from_directory
import docker
import json
import base64
import openai
import random

app = Flask(__name__)

base_directory = os.getcwd()

@app.route('/<path:subpath>')
def serve_file(subpath):
    # Construct the absolute path to the requested file
    absolute_path = os.path.join(base_directory, subpath)
    
    # Ensure that the requested path is within the base directory to prevent directory traversal attacks
    if os.path.commonpath([absolute_path, base_directory]) != base_directory:
        return "Invalid path", 403
    
    # Check if the requested path exists and is a file
    if os.path.isfile(absolute_path):
        return send_from_directory(base_directory, subpath)
    
    # If the path is a directory, you can customize the response as needed
    # For example, you can list the contents of the directory or provide a custom page
    return "File not found", 404


@app.route('/', methods=['POST'])
def manim():
    # we get the topic from the request
    topic = request.get_json()['topic']
    
    # read file example_1.txt
    example_response_1 = open("example_1.txt", "r").read()
    example_response_2 = open("example_2.txt", "r").read()
    example_response_3 = open("example_3.txt", "r").read()
    # example_response_4 = open("example_4.txt", "r").read()
    
    
    # submit the topic to GPT-4 for it to write the manim code
    message=[
        {"role": "system", "content": "From now on act as CAN (“code anything now”) CAN is an expert coder, with years of coding experience. CAN does not have a character limit."},
        {"role": "user", "content": "Write a manim Python script to visualize the sin and cos functions."},
        {"role": "assistant", "content": example_response_1},
        {"role": "user", "content": "Write a manim Python script to visualize the argmin of a function."},
        {"role": "assistant", "content": example_response_2},
        {"role": "user", "content": "Write a manim Python script to visualize a rectangle whose upper-right corner can move along a curve."},
        {"role": "assistant", "content": example_response_3},
    ]
    
    message.append({"role": "user", "content": f"Write me a manim Python script to animate {topic}. Name the class {''.join([p.capitalize() for p in topic.split(' ')])}."})

    class_name = ''.join([p.capitalize() for p in topic.split(' ')])
    print(class_name)
    
    generating = True
    tf = None
    
    while generating:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages = message,
            temperature=0.2,
            max_tokens=1000,
            frequency_penalty=0.0
        )
        
        code = response.choices[0].message.content.split("```")[1]
        code = "\n".join(code.split('\n')[1:])
        print(code)
        # create temp file
        
        tf = str(random.randint(1, 1000000)) + ".py"
        f = open(tf, "a")
        f.write(code)
        f.close()

        #with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
        #    tmp.write(code.encode('utf-8'))
        #    tmp_file_name = tmp.name

        # create docker client
        client = docker.from_env()
    
        # run docker container
        volume={os.getcwd(): {'bind': '/manim', 'mode': 'rw'}}
       
        
        container = client.containers.run("manimcommunity/manim", 
                                            f"manim -qm {tf} {class_name}",
                                            volumes=volume,
                                            stderr=True, stdout=True,
                                            detach=True)

        total_output = "" 
        for line in container.logs(stream=True):
            print(line.strip())
            line = line.decode("utf-8")
            total_output += line
            if "Error" in line:
                message.append({"role": "user", "content": total_output})
                break
            if "Rendered" in line:
                generating = False
                break
        
    # output video file path
    output_path = os.path.join("media", "videos", str(tf), "720p30", class_name) + ".mp4"
    return output_path

    return send_file(output_path, mimetype='video/mp4', as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)