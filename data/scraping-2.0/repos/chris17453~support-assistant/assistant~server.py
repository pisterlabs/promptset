import os
import json
import uuid
import openai
from flask import Flask, render_template, Response, send_from_directory, jsonify,request, session
from flask_jwt_extended import create_access_token, jwt_required
from flask import Blueprint
from . import process
from . import audio
from . import image
from . import video
from . import avatar
from . import settings

# Create a blueprint for the server routes
server_blueprint = Blueprint("server", __name__)




openai.api_key = settings.openai_api_secret


@server_blueprint.route('/')
def index():
    """Video streaming home page."""
    options=[]
    options_base=avatar.get_by_link_id(1)

    # give a default avatar if none is selected
    if 'avatar' not in session:
        session['avatar']=options_base[0].uuid

    img_uuid=get_avatar_poster(session['avatar'])

    #for option in options_base:
    #    options.append({option.name,option.uuid})

    #print(options)
    return render_template('index.html',options=options_base, img_uuid=img_uuid)




@server_blueprint.route('/js/<path:path>')
def serve_js(path):
    return send_from_directory('static/js', path)

@server_blueprint.route('/js/vanta/<path:path>')
def serve_js2(path):
    return send_from_directory('static/js/vanta', path)

@server_blueprint.route('/css/<path:path>')
def serve_css(path):
    return send_from_directory('static/css', path)

@server_blueprint.route('/images/<path:path>')
def images(path):
    """Background image for video transitions."""
    return send_from_directory('../assets/images/',path)

@server_blueprint.route('/ui/<path:path>')
def ui(path):
    """Background image for video transitions."""
    return send_from_directory('static/ui/',path)


def gen(video_file):
    """Video streaming generator function."""


    if not os.path.isfile(video_file):
        video_file=os.path.basename(video_file) 
        current_directory = os.getcwd()
        video_file=os.path.join(current_directory,"assets","video",video_file)

    with open(video_file, 'rb') as f:
        yield from f

@server_blueprint.route('/audio/<path:path>', methods=['GET'])
def audio_feed(path):
    """Audio streaming route. Put this in the src attribute of an img tag."""
    print("Looking for Audio UUID: {0}".format(path))
    record=audio.get_by_uuid(path)
    if record==None:
        return "Not Found", 404 

    return Response(gen(record.path+"/"+record.name),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@server_blueprint.route('/video/<path:path>', methods=['GET'])
def video_feed(path):
    """Video streaming route. Put this in the src attribute of an img tag."""
    print("Looking for Video UUID: {0}".format(path))
    record=video.get_by_uuid(path)
    if record==None:
        return "Not Found", 404 

    return Response(gen(record.path+"/"+record.name),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@server_blueprint.route('/image/<path:path>', methods=['GET'])
def image_feed(path):
    """Image route. Put this in the src attribute of an img tag."""
    print("Looking for Image UUID: {0}".format(path))
    record=image.get_by_uuid(path)
    if record==None:
        return "Not Found", 404 

    return send_from_directory(settings.image_directory,record.image_name)
                    



@server_blueprint.route('/access_token/', methods=['POST'])
def access_token():
    # Generate JWT token
    session_id=create_session()
    access_token = create_access_token(session_id)
    session['session_id'] = session_id


    
    return jsonify({'access_token': access_token}), 200
    session_id = session.get('session_id')

def create_session():
    session_id = str(uuid.uuid4())

    # Create the sessions directory if it doesn't exist
    if not os.path.exists(settings.sessions_directory):
        os.makedirs(settings.sessions_directory)

    message=open(os.path.join(settings.persona_directory,'chrisbot.txt'), 'r').read()
    add_to_session(session_id, 'system', message)
    
    # Define the file path
    file_path = os.path.join(settings.sessions_directory, session_id)

    # Define the assistant tuning prompt
    # read a file for the tuning prompt
     
    return session_id

def add_to_session(session_id, role, message):
    message = {'role':role,'content':message }
    file_path=os.path.join(settings.sessions_directory, session_id)
    
    if os.path.exists(file_path):
        # Open the file in read mode
        with open(file_path, 'r') as file:
            existing_data = json.load(file)

        # Update the existing data with new content
        existing_data.append(message)
    else:
        existing_data=[message]

    # Open the file in write mode
    with open(file_path, 'w') as file:
        json.dump(existing_data, file)

        
def get_session(session_id):
    file_path=os.path.join(settings.sessions_directory, session_id)
    # Open the file in read mode
    with open(file_path, 'r') as file:
        existing_data = json.load(file)
    return existing_data


def process_data(session_id,data):
    
    add_to_session(session_id, 'user', data)
    history=get_session(session_id)


    chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=history
        )
    reply = chat.choices[0].message.content
    avatar_rec=avatar.get_by_uuid(session['avatar'])
    if avatar_rec==None:
        return {'message': 'avatar failed to load'}, 500
    print ("Using Avatar {0}:{1}".format(avatar_rec.name,avatar_rec.id))
    utterance_obj=process.utterance_builder(avatar_rec.id,reply,True)
    audio_uuid=utterance_obj.audio.uuid
    video_uuid=utterance_obj.video.uuid

    add_to_session(session_id, 'assistant', reply)

    return {'message': 'Authorized','role':'assistant','content':reply,'video':video_uuid,'audio':audio_uuid}


@jwt_required
@server_blueprint.route('/api/talk', methods=['POST'])
def talk():
    session_id=session['session_id']
    data = request.get_json()
    # Process the data
    result = process_data(session_id,data)  
    print(data[-1])
    return jsonify(result) 


def get_avatar_poster(avatar_uuid):
    av=avatar.get_by_uuid(avatar_uuid)
    if av:
        img=image.get_by_id(av.image_id)
        return img.uuid
    else:
        return None


@jwt_required
@server_blueprint.route('/api/avatar', methods=['POST'])
def avatart_image():
    session_id=session['session_id']
    data = request.get_json()
    # Process the data
    print(data)
    session['avatar']=data['avatar']
    img_uuid=get_avatar_poster(data['avatar'])
    result = {'message': 'Authorized','poster':img_uuid}
    return jsonify(result) 







