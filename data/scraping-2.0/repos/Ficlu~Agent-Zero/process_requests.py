from flask import request, jsonify, Blueprint, session
from config import openai
from datetime import date
import os
from input_router import InputRouter
from agent_profile import AgentProfile
from user_profile import UserProfile
process_requests = Blueprint('process_requests', __name__)

@process_requests.route('/process', methods=['POST'])
def process():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 403  # HTTP status code 403 means "Forbidden"
    
    for key, value in request.form.items():
        print(f"{key}: {value}")
    user_input = request.form.get('message')
    user_id = session.get('username')  # Get the username from the session
    agent_id = request.form.get('agent_id')
    chat_id = request.form.get('chat_id')
    if not all([user_input, user_id, agent_id, chat_id]):
        return jsonify({'error': 'Missing required fields'}), 400

    agent_id = int(agent_id)
    chat_id = int(chat_id)
    print("user id: " + user_id)
    response = InputRouter().route(user_id, agent_id, chat_id, 1, date.today().isoformat(), user_input)
    print(response)
    return jsonify(response)


@process_requests.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 403

    audio_file = request.files['audio']
    # Save the audio file temporarily and open it for reading
    audio_file.save("temp_audio.wav")
    with open("temp_audio.wav", "rb") as f:
        transcript = openai.Audio.transcribe("whisper-1", f)

    # Remove the temporary file
    os.remove("temp_audio.wav")
    
    return jsonify({'transcription': transcript.text})


@process_requests.route('/create_agent', methods=['POST'])
def create_agent():
    
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 403  # HTTP status code 403 means "Forbidden"
    user_id = session['username']
    # Get the JSON data from the request
    data = request.get_json()
    print(f"ln 50 process_requests: {data}")


    # Extract each field from the request
    agent_name = data.get('agent_name')
    agent_type = data.get('agent_type')
    agent_gender = data.get('agent_gender')
    agent_ethnicity = data.get('agent_ethnicity')
    agent_dob = str(data.get('agent_dob'))
    agent_physical_characteristic = data.get('agent_physical_characteristic')
    agent_relationship = data.get('agent_relationship')
    agent_personality = data.get('agent_personality')
    agent_IQ = int(data.get('agent_IQ'))
    agent_EQ = int(data.get('agent_EQ'))
    agent_voice = data.get('agent_voice')

    # Check that all required fields are present
    if not all([agent_name, agent_type, agent_gender, agent_ethnicity, agent_dob, agent_physical_characteristic, agent_relationship, agent_personality, agent_IQ, agent_EQ, agent_voice]):
        return jsonify({'error': 'Missing required fields'}), 400  # HTTP status code 400 means "Bad Request"

    # Create a new agent

    agent_details = {
        "agent_type": agent_type,
        "agent_gender": agent_gender,
        "agent_ethnicity": agent_ethnicity,
        "agent_dob": agent_dob,
        "agent_physical_characteristic": agent_physical_characteristic,
        "agent_relationship": agent_relationship,
        "agent_personality": agent_personality,
        "agent_IQ": agent_IQ,
        "agent_EQ": agent_EQ
    }
    new_agent_profile = AgentProfile(user_id, agent_name, agent_voice, agent_details)
    # Save the new agent (you will need to implement this according to your specific database/ORM)
    try:
        
        agent_id = new_agent_profile.save()
    except Exception as e:
        print(e)  # print the exception details
        return jsonify({'error': str(e)}), 500

    return jsonify({'message': 'Agent created successfully', 'status': 'success', 'agent_id': agent_id}), 201

@process_requests.route('/create_user_profile', methods=['POST'])
def create_user_profile():
    
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 403  # HTTP status code 403 means "Forbidden"
    user_id = session['username']
    # Get the JSON data from the request
    data = request.get_json()
    print(f"ln 50 process_requests: {data}")

    # Extract each field from the request
    u_name = data.get('u_name')
    u_age = int(data.get('u_age'))
    u_ethnicity = data.get('u_ethnicity')
    u_gender = data.get('u_gender')
    u_personality = data.get('u_personality')
    u_iq = int(data.get('u_iq'))
    u_labels = data.get('u_labels')

    print(f'u_name: {u_name}, u_age: {u_age}, u_ethnicity: {u_ethnicity}, u_gender: {u_gender}, u_personality: {u_personality}, u_iq: {u_iq}, u_labels: {u_labels}')

    # Check that all required fields are present
    if not all([u_name, u_age, u_ethnicity, u_gender, u_personality, u_iq, u_labels]):
        return jsonify({'error': 'Missing required fields'}), 400  # HTTP status code 400 means "Bad Request"

    # Create a new user profile

    user_profile_details = {
        "u_age": u_age,
        "u_ethnicity": u_ethnicity,
        "u_gender": u_gender,
        "u_personality": u_personality,
        "u_iq": u_iq,
        "u_labels": u_labels
    }
    new_user_profile = UserProfile(user_id, u_name, user_profile_details)
    # Save the new user profile (you will need to implement this according to your specific database/ORM)
    try:
        user_id = new_user_profile.save()
    except Exception as e:
        print(e)  # print the exception details
        return jsonify({'error': str(e)}), 500

    return jsonify({'message': 'User profile created successfully', 'status': 'success', 'user_id': user_id}), 201
