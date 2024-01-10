from flask import current_app as app, render_template, jsonify, request,g
import os, json
import subprocess
from werkzeug.security import check_password_hash, generate_password_hash
from application.models import db, User, UserTranscription,UserRoles
from flask_security import  current_user,auth_required
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import spacy
from collections import Counter
from openai import OpenAI
from application.email import send_email_user
from jinja2 import Template
import random
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
# Set the OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) # Initialize the OpenAI client
nlp = spacy.load("en_core_web_sm") # Load the spaCy model


# Define the home page route
@app.route('/')
def index():
    return render_template('index.html')

# Define the dictionary of user information
def cuser_to_dict(user):
    return {
        'id': user.id,
        'username': user.username,
        'email': user.email,
    }

# Define the dictionary of user information
def puser_to_dict(user):
    return {
        'id': user.id,
        'username': user.username,
        'email': user.email,
        'image': user.image,
    }

# Define the dictionary of usertranscription information
def transcript_to_dict(transcript):
    return {
        'id': transcript.id,
        'text': transcript.transcription,
        'language': transcript.language,
        'user_id': transcript.user_id,
        'created_on': transcript.time,
    }


# Define the route for user deletion
@app.route('/deleteuser/', methods=['DELETE'])
@jwt_required()
def deleteuser():
    id = get_jwt_identity()
    user=User.query.filter_by(id=id).first()
    if not user:
        return jsonify({'message': 'No user found!'})
    usertranscript=UserTranscription.query.filter_by(user_id=id).all()
    role = UserRoles.query.filter_by(user_id=id).first()
    db.session.delete(role)
    db.session.commit()
    for i in usertranscript:
        db.session.delete(i)
        db.session.commit()
    db.session.delete(user)
    db.session.commit()
    return jsonify({'message': 'User deleted successfully!'})


# Define the route for user resetpassword
@app.route('/resetpassword', methods=['PUT','POST'])
def resetpassword():
    if request.method=='POST':
        post_data = request.get_json()
        email = post_data.get('email')
        user = User.query.filter_by(email=email).first()
        genotp= random.randint(100000,999999) 
        if not user:
            return jsonify({'message': 'No user found!'})
        with open('templates/reset.html') as file_:
            template = Template(file_.read())
            message = template.render(otp=genotp)

        send_email_user(
            to=email,
            sub="Password Reset",
            message=message
        )

        return jsonify({'message': 'Password sent successfully!', 'otp': genotp, 'email': email})
    
    if request.method=='PUT':
        post_data = request.get_json()
        email = post_data.get('email')
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({'message': 'No user found!'})
        password = generate_password_hash(post_data.get('password'))
        user.password=password
        db.session.commit()
        return jsonify({'message': 'Password reset successfully!'})



# Define the route for user login
@app.route('/userlogin', methods=['POST'])
def userlogin():
    post_data = request.get_json()
    username = post_data.get('username')
    password = post_data.get('password')

    with app.app_context():
        user_datastore = app.security.datastore
        user = User.query.filter_by(username=username).first()

        if not user:
            app.logger.info(f"No user found for username: {username}")
            return jsonify({'message': 'No user found!'})

        if check_password_hash(user.password, password):
            app.logger.info("Password validation successful")
            access_token = create_access_token(identity=user.id)
            return jsonify({"token": access_token})
        else:
            app.logger.warning("Password validation failed")
            return jsonify({"message": "Wrong Password"})


# Define the route for user profile
@app.route("/userprofile/", methods=['POST','PUT','GET'])
@jwt_required()
def userprofile():
    id = get_jwt_identity()
    if request.method=='GET':
        user=User.query.filter_by(id=id).first()
        return jsonify(puser_to_dict(user))
    if request.method=='PUT':
        post_data = request.get_json()
        image = post_data.get('image')
        password = post_data.get('password')
        user=User.query.filter_by(id=id).first()
        if not user:
            return jsonify({'message': 'No user logged in'})
        if image:
            user.image=image
            db.session.commit()
        if password:
            user.password=generate_password_hash(password)
            db.session.commit()
        return jsonify({'message': 'User updated successfully!'})

# Define the route for currentuser
@app.route('/currentuser/')
@jwt_required()
def currentuser():
    user=User.query.filter_by(id=get_jwt_identity()).first()
    if not user:
        return jsonify({'message': 'No user logged in'})
    return jsonify(cuser_to_dict(user))


# Define the route for user creation and listing
@app.route('/createuser/')
def createuser():
    user=User.query.all()
    return jsonify([cuser_to_dict(user) for user in user])


# Define the route for user creation
@app.route('/registeruser/', methods=['POST'])
def registeruser():
    post_data = request.get_json()
    username = post_data.get('username')
    email = post_data.get('email')
    password = post_data.get('password')
    image = post_data.get('image')
    if not username:
        return jsonify({'message': 'Username is required'})
    if not email:
        return jsonify({'message': 'Email is required'})
    if not password:
        return jsonify({'message': 'Password is required'})
    user = User.query.filter_by(username=username,email=email).first()
    if user:
        return jsonify({'message': 'Username already exists'})
    with app.app_context():
        user_datastore = app.security.datastore
        if not user_datastore.find_user(username=username) and not user_datastore.find_user(email=email):
            user_datastore.create_user(username=username, email=email,image=image, password=generate_password_hash(password))
            db.session.commit()
            user = user_datastore.find_user(username=username)
            role = user_datastore.find_role('user')
            user_datastore.add_role_to_user(user, role)
            db.session.commit()

    return jsonify({'message': 'User created successfully!'})

# Define the route for usertanscription
@app.route('/usertranscript/')
@jwt_required()
def usertranscript():
    user=UserTranscription.query.filter_by(user_id=get_jwt_identity()).order_by(UserTranscription.time.desc()).limit(30)
    return jsonify([transcript_to_dict(user) for user in user])

# Define the route for usertanscriptionanalysis
@app.route('/usertranscriptanalysis/')
@jwt_required()
def compute_frequent_words_and_phrases():
    user_id = get_jwt_identity()

    # Calculate the most frequently used words for the current user
    user_transcriptions = UserTranscription.query.filter_by(user_id=user_id).all()
    all_transcriptions = " ".join([transcription.transcription for transcription in user_transcriptions])
    doc = nlp(all_transcriptions)
    frequent_words = [token.text for token in doc if token.is_alpha and not token.is_stop]
    frequent_words_counter = Counter(frequent_words)
    frequent_words_user = dict(frequent_words_counter.most_common(10))  # Adjust the number as needed

    # Calculate the most frequently used words across all users
    all_transcriptions = " ".join([transcription.transcription for transcription in UserTranscription.query.all()])
    doc_all_users = nlp(all_transcriptions)
    frequent_words_all_users = Counter([token.text for token in doc_all_users if token.is_alpha and not token.is_stop])
    frequent_words_all_users = dict(frequent_words_all_users.most_common(10))  # Adjust the number as needed

    return jsonify({'frequent_words_user': frequent_words_user, 'frequent_words_all_users': frequent_words_all_users})

# Define the route for useruniquephrases
@app.route('/useruniquephrases/')
@jwt_required()
def get_user_unique_phrases():
    user_id = get_jwt_identity()

    # Retrieve all transcriptions for the current user
    user_transcriptions = UserTranscription.query.filter_by(user_id=user_id).all()

    # Extract and count phrases from the transcriptions
    all_phrases = []
    for transcription in user_transcriptions:
        phrases = extract_phrases(transcription.transcription)
        all_phrases.extend(phrases)

    # Count the frequency of each phrase
    phrase_counts = Counter(all_phrases)

    # Extract unique phrases used only once
    unique_phrases = [phrase for phrase, count in phrase_counts.items() if count == 1]

    # Return the first 3 unique phrases (or all if there are fewer than 3)
    return jsonify({'user_unique_phrases': unique_phrases[:3]})

def extract_phrases(text):
    # You can customize this function based on your requirements for extracting phrases
    doc = nlp(text)
    phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) >= 2]
    return phrases


# Define the route for similarusers

@app.route('/similarusers/')
@jwt_required()
def find_similar_users():
    current_user_id = get_jwt_identity()

    # Retrieve transcriptions for the current user
    current_user_transcriptions = UserTranscription.query.filter_by(user_id=current_user_id).all()

    if len(current_user_transcriptions) == 0:
        return jsonify({'similar_users': []})

    # Extract text from transcriptions
    current_user_text = " ".join([transcription.transcription for transcription in current_user_transcriptions])

    # Retrieve transcriptions for all users (excluding the current user)
    all_users_transcriptions = UserTranscription.query.filter(UserTranscription.user_id != current_user_id).all()

    if len(all_users_transcriptions) == 0:
        return jsonify({'similar_users': []})

    # Create a list of user texts
    all_users_texts = [" ".join([transcription.transcription for transcription in UserTranscription.query.filter_by(user_id=user_transcription.user_id).all()]) for user_transcription in all_users_transcriptions]

    # Calculate TF-IDF vectors for the current user and all users
    vectorizer = TfidfVectorizer()
    current_user_vector = vectorizer.fit_transform([current_user_text])
    all_users_vectors = vectorizer.transform(all_users_texts)

    # Calculate cosine similarity between the current user and all users
    similarities = cosine_similarity(current_user_vector, all_users_vectors)[0]

    # Get the indices of users with the highest similarity
    most_similar_user_indices = similarities.argsort()[:-4:-1]  # Get top 3 most similar users

    # Retrieve user information for the most similar users
    most_similar_users = [User.query.get(all_users_transcriptions[i].user_id) for i in most_similar_user_indices]

    # Convert user information to a dictionary format
    similar_users_info = []
    for i in range(len(most_similar_users)):
        if len(similar_users_info)==5:
            break
        if most_similar_users[i].username != User.query.get(current_user_id).username:
            similar_users_info.append(most_similar_users[i].username)

    similar_users_info=list(set(similar_users_info))

    return jsonify({'similar_users': similar_users_info})


# Define the route for speech to text conversion
@app.route('/speech/<lang>', methods=['POST'])
def speech(lang):
    user_id = request.form.get('user_id')
    audio_file = request.files['audio']

    # Create the directory if it doesn't exist
    audio_dir = os.path.join(app.root_path, 'static', 'js', 'audio')
    os.makedirs(audio_dir, exist_ok=True)

    # Save the audio file to a known location with Ogg extension
    audio_file_path = os.path.join(audio_dir, 'audio.ogg')
    audio_file.save(audio_file_path)
    audio_file_size_bytes = os.path.getsize(audio_file_path)
    
    # Convert the file size to MB
    audio_file_size_mb = audio_file_size_bytes / (1024 * 1024)
    # Check if the file size is larger than 25 MB
    if audio_file_size_mb > 5:
        return jsonify({'text': 'File size is larger than 5 MB'})
    audio_file_open = open(audio_file_path, "rb")
    try:
        if lang=="English":
            transcript = client.audio.translations.create(
            model="whisper-1", 
            file=audio_file_open, 
            response_format="json",
            prompt="i am talking in"+lang
            )
        elif lang=="":
            transcript = client.audio.translations.create(
            model="whisper-1", 
            file=audio_file_open, 
            response_format="json"
            )
        else:
            transcript = client.audio.translations.create(
            model="whisper-1", 
            file=audio_file_open, 
            response_format="json",
            prompt="i am talking in"+lang
            )

        if user_id!='':
            user_transcription = UserTranscription(user_id=user_id, transcription=transcript.text, language=lang, time=datetime.datetime.now())
            db.session.add(user_transcription)
            db.session.commit()
        return jsonify({'text': transcript.text})
    except Exception as e:
        print(e)
        return jsonify({'text': 'Error in transcription'})
    finally:
        audio_file_open.close()
        os.remove(audio_file_path)
