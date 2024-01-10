import os
import openai
import pandas as pd
import numpy as np
import hashlib
import time
import threading
import csv
import json
import re
import ast
import shutil

from csv import reader
from app import app, login_manager
from flask import render_template, flash, redirect, url_for, request, session, jsonify

#Secure against CSRF attacks
from flask_wtf.csrf import generate_csrf
from flask_wtf.csrf import CSRFProtect
from flask_login import login_required, current_user, UserMixin
from flask_login import login_user, logout_user, login_required
from werkzeug.utils import secure_filename
from app.forms import UploadSyllabus
from app.routes_helper import (
    save_syllabus, 
    delete_previous_syllabus,
    check_processed_files,
    detect_final_data_files,
    load_course_metadata
    #courses_with_final_data, # Checks all (courses) sub-folders returns a list that contain both 'textchunks.npy' and 'textchunks-originaltext.csv' files.
    #retry_with_exponential_backoff # Define a retry decorator with exponential backoff
)
from app.chop_documents import chunk_documents_given_course_name
#from app.embed_documents import embed_documents_given_course_name
from app.embed_documents_v2 import embed_documents_given_course_name
from app.create_final_data import create_final_data_given_course_name
from app.file_operations import (
    read_from_file_json,
    write_to_file_json,
    read_from_file_text,
    check_folder_exists,
    list_folders,
    rename_folder,
    delete_folder,
    create_course_folder_with_metadata,
    save_course_metadata,
    allowed_file,
    delete_file,
    get_first_txt_file,
    get_file_path,
    delete_files_in_folder,
    get_content_files,
    check_and_update_activations_file,
    read_csv_preview
)

from pdfminer.high_level import extract_text
from io import BytesIO
from urllib.parse import urlparse


# -------------------- Global Variables --------------------
# this lets us load the data only once and to do it in the background while the user types the first q
df_chunks = None
embedding = None
last_session = None

# define the variable to show / hide all the Courses available to the system


SETTINGS_PATH = os.path.join(app.config['FOLDER_SETTINGS'], 'platform-settings.json')

# Load custom settings from JSON
custom_settings = read_from_file_json(SETTINGS_PATH)
for key, value in custom_settings.items():
    app.config[key.upper()] = True if value["Value"] == "True" else False


#Flask function to escape special characters:
def escape_id(id_str):
    return id_str.replace(".", "\.").replace(":", "\:")

app.jinja_env.globals.update(escape_id=escape_id)

@app.template_filter('strip_domain')
def strip_domain_filter(url):
    if not isinstance(url, str):
        return url  # Return the original value if it's not a string

    parsed = urlparse(url)
    return parsed.path + ('?' + parsed.query if parsed.query else '')
app.jinja_env.filters['strip_domain'] = strip_domain_filter


def format_json(message):
    try:
        parsed_json = json.loads(message)
        return json.dumps(parsed_json, indent=4)
    except json.JSONDecodeError:
        return message
    
@app.template_filter('break_lines')
def break_lines(s):
    return "<br>".join([s[i:i+50] for i in range(0, len(s), 50)])

app.jinja_env.filters['break_lines'] = break_lines

# -------------------- Flask app configurations --------------------

openai.api_key = os.getenv("OPENAI_API_KEY")


# -------------------- Flask Overrides  --------------------


# Flask Login Overrides to load additional attributes
class User(UserMixin):
    def __init__(self, id=None, folder=None, admin=None):
        self.id = id
        self.folder = folder
        self.admin = admin


# Used to not have chache on pages which effected the processing of information in the background
@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r



# -------------------- Basic Admin Authentication --------------------

#Define the Username and Password to access the Logs
users = read_from_file_json(os.path.join(app.config['FOLDER_SETTINGS'], "users.json")) 

## Debug
# print(users)

#class User(UserMixin):
#  pass

#Define the Username and Password to access the Logs and Debugging  
@login_manager.user_loader
def user_loader(username):
  if username not in users:
    return None
  #user = User()
  #user.id = username
  user_data = users[username]
  user = User(id=username, folder=user_data['folder'], admin=user_data['admin'])
  return user

@login_manager.request_loader
def request_loader(request):
  username = request.form.get('username')
  if username not in users:
    return None
  user_data = users[username]
  user = User(id=username, folder=user_data['folder'], admin=user_data['admin'])
  # user.is_authenticated = request.form['pw'] == user_data['pw']
  return user
    



# -------------------- Routes --------------------
@app.route('/')
@app.route('/index')
def index():
  app.logger.info(f"Loading Index Page")
  return redirect(url_for('adminlogin'))
  #return render_template('index.html')
  print(app.config["SHOWALLCOURSESAVAILABLE"])
  return render_template('index.html', name=session.get('name'), showAllCoursesAvailable=app.config["SHOWALLCOURSESAVAILABLE"])

@app.route('/privacy-policy')
def privacypolicy():
    return render_template('privacy.html')

#  --------------------Routes for the login and logout pages --------------------
@app.route('/admin-login', methods=['GET', 'POST'])
def adminlogin():
    if session.get('name'):
        return redirect(url_for('courses'))
    else:
        if request.method == 'POST':
            username = request.form.get('username')
            user_data = users.get(username, {})
            if request.form.get('pw') == users.get(username, {}).get('pw'):
                #user = User()
                user = User(id=username, folder=user_data.get('folder'), admin=user_data.get('admin'))
                login_user(user)
                session['name'] = user.id
                session['folder'] = user.folder
                session['admin'] = user.admin
                #app.config['FOLDER_UPLOAD'] = os.path.join(app.config['FOLDER_UPLOAD'], session['folder'])
                app.logger.info(f"User {session['name']} logged in successfully, folder: {session['folder']}, admin: {session['admin']}")
                return redirect(url_for('courses'))
            else:
                flash('Incorrect username or password!', 'error')
    return render_template('adminlogin.html', name=session.get('name'), folder=session.get('folder'), admin=session.get('admin'))
  
@app.route('/logout')
def logout():
  logout_user()
  session.clear()  # Clear session data
  return redirect(url_for('adminlogin'))

#  --------------------Routes for Course Management --------------------

@app.route('/courses', methods=['GET'])
@login_required
def courses():
    courses = list_folders()
    return render_template('courses.html', courses=courses, name=session.get('name'), folder=session.get('folder'), admin=session.get('admin'))

@app.route('/create-course', methods=['POST'])
@login_required
def create_course():
    name = request.form['coursename'].replace(' ', '-')
    metadata = {
        'classname': name,
        'professor': request.form.get('professor', ''),
        'assistants': request.form.get('assistants', ''),
        'classdescription': request.form.get('classdescription', ''),
        'assistant_name': request.form.get('assistant_name', '')
    }    
    create_course_folder_with_metadata(name, metadata)
    return redirect(url_for('courses'))

@app.route('/rename-item', methods=['POST'])
@login_required
def rename_item():
    old_name = request.form['old_name']
    new_name_raw = request.form['new_name'].replace(' ', '-')
    course_name = request.form.get('course_name', None)
    user_folder = session['folder']

    # Extract the extension from the old filename
    old_extension = os.path.splitext(old_name)[1]

    # Check if the new name has the same extension; if not, add the old extension to it
    if not new_name_raw.lower().endswith(old_extension.lower()):
        new_name = new_name_raw + old_extension
    else:
        new_name = new_name_raw

    # Secure the new filename
    new_name = secure_filename(new_name)

    if course_name:
        # This is a file (content) within a course (folder)
        old_path = os.path.join(app.config["FOLDER_PREUPLOAD"], user_folder, course_name, old_name)
        new_path = os.path.join(app.config["FOLDER_PREUPLOAD"], user_folder, course_name, new_name)
        if os.path.exists(old_path):
            rename_folder(old_path, new_path) # Replace with the appropriate function to rename a file or folder
    else:
        # This is a course (folder)
        # First rename the Course Folder in the "Upload" Section
        old_path_upload = os.path.join(app.config["FOLDER_UPLOAD"], user_folder, old_name)
        new_path_upload = os.path.join(app.config["FOLDER_UPLOAD"], user_folder, new_name)
        if os.path.exists(old_path_upload):
            rename_folder(old_path_upload, new_path_upload)
        old_path_upload = os.path.join(app.config["FOLDER_PREUPLOAD"], user_folder, old_name)
        new_path_upload = os.path.join(app.config["FOLDER_PREUPLOAD"], user_folder, new_name)
        if os.path.exists(old_path_upload):
            rename_folder(old_path_upload, new_path_upload)
        #check if there is a syllabus file and rename it to the new cousename
        old_syllabus_path = os.path.join(app.config["FOLDER_UPLOAD"], user_folder, new_name, "Syllabus-" + old_name + ".pdf")
        new_syllabus_path = os.path.join(app.config["FOLDER_UPLOAD"], user_folder, new_name, "Syllabus-" + new_name + ".pdf")
        if os.path.exists(old_syllabus_path):
            rename_folder(old_syllabus_path, new_syllabus_path)
            app.logger.info(f"Renamed Syllabus file from {old_syllabus_path} to {new_syllabus_path}")
            #Now we check if there were chunked and embedded files for the old course name and delete them
            old_syllabus_chunk_folder_path = os.path.join(app.config["FOLDER_UPLOAD"], user_folder, new_name, 'Textchunks',"Syllabus-" + old_name + "-originaltext.csv")
            old_syllabus_embedded_folder_path = os.path.join(app.config["FOLDER_UPLOAD"], user_folder, new_name, 'EmbeddedText',"Syllabus-" + old_name + "-originaltext.npy")
            if os.path.exists(old_syllabus_chunk_folder_path):
                delete_file(old_syllabus_chunk_folder_path)
                app.logger.info(f"Deleted Syllabus chunk file: {old_syllabus_chunk_folder_path}")
            if os.path.exists(old_syllabus_embedded_folder_path):
                delete_file(old_syllabus_embedded_folder_path)
                app.logger.info(f"Deleted Syllabus embedded file: {old_syllabus_embedded_folder_path}")
            # We run the chunking, embedding and creating final data for the newly renamed course name
            # This is needed as we renamed the course, so the Syllabus file name has changed
            chunk_documents_given_course_name(os.path.join(app.config['FOLDER_UPLOAD'], user_folder, new_name))
            app.logger.info(f"Completed chop_course_content: COURSE NAME: {new_name}")
            embed_documents_given_course_name(os.path.join(app.config['FOLDER_UPLOAD'], user_folder, new_name))
            app.logger.info(f"Completed embed_course_content")       
            app.logger.info(f"Creating final data for course: COURSE NAME: {new_name}")
            create_final_data_given_course_name(os.path.join(app.config['FOLDER_UPLOAD'], user_folder, new_name))
            app.logger.info(f" Completed final data for course: COURSE NAME: {new_name}")

    return redirect(request.referrer)

@app.route('/delete-item', methods=['GET'])
@login_required
def delete_item():
    name = request.args.get('name')
    #$print(name)
    course_name = request.args.get('course_name', None)
    user_folder = session['folder']
    if course_name: # So this is a deletion of a single file
        # This is a file (content) within a course (folder)
        path = os.path.join(app.config["FOLDER_UPLOAD"], user_folder, course_name, name)
        if os.path.exists(path):
            delete_file(path)
            app.logger.info(f"Deleted file: {path}")
            #We check if this file was chunked and embedded and delete those files as well
            chunk_folder_path = os.path.join(app.config["FOLDER_UPLOAD"], user_folder, course_name, 'Textchunks',name.split('.')[0] + "-originaltext.csv")
            embedded_folder_path = os.path.join(app.config["FOLDER_UPLOAD"], user_folder, course_name, 'EmbeddedText',name.split('.')[0] + "-originaltext.npy")
            if os.path.exists(chunk_folder_path):
                delete_file(chunk_folder_path)
                app.logger.info(f"Deleted chunk file: {chunk_folder_path}")
            if os.path.exists(embedded_folder_path):
                delete_file(embedded_folder_path)
                app.logger.info(f"Deleted embedded file: {embedded_folder_path}")
            #Now that we have deleted the file, we need to run the chunking, embedding and creating final data in the background
            #this is needed as we deleted a file, so the final data needs to be updated should the deleted file have been part of the final data
            chunk_documents_given_course_name(os.path.join(app.config['FOLDER_UPLOAD'], user_folder, course_name))
            app.logger.info(f"Completed chop_course_content")
            embed_documents_given_course_name(os.path.join(app.config['FOLDER_UPLOAD'], user_folder, course_name))
            app.logger.info(f"Completed embed_course_content")
            app.logger.info(f"Creating final data for course: COURSE NAME: {course_name}")
            create_final_data_given_course_name(os.path.join(app.config['FOLDER_UPLOAD'], user_folder, course_name))
            app.logger.info(f" Completed final data for course: COURSE NAME: {course_name}")
        path = os.path.join(app.config["FOLDER_PREUPLOAD"], user_folder, course_name, name)
        if os.path.exists(path):
            delete_file(path)
            app.logger.info(f"Deleted file: {path}")            
    else: # So this is a deletion of a folder
        # This is a course (folder)
        # Delete all the Corresponding Course folders in all places
        path = os.path.join(app.config["FOLDER_UPLOAD"], user_folder, name)
        if os.path.exists(path):
            delete_folder(path)
            app.logger.info(f"Deleted folder: {path}")
        path = os.path.join(app.config["FOLDER_PREUPLOAD"], user_folder, name)
        if os.path.exists(path):
            delete_folder(path)
            app.logger.info(f"Deleted folder: {path}")
    return redirect(request.referrer)

@app.errorhandler(400)
def bad_request_error(error):
    app.logger.error(f"Bad Request Error: {error}")
    return jsonify(success=False, error="Bad Request"), 400

@app.route('/toggle_activation/<course_name>/<file_name>', methods=['POST'])
@login_required
def toggle_activation(course_name, file_name):
    app.logger.info(f"Raw request values: {request.values}")
    app.logger.info(f"Entered toggle_activation for course {course_name} and file {file_name}")    
    try:
        app.logger.info("Starting toggle_activation logic...")
        app.logger.info(f"Entering toggle_activation with course_name: {course_name}, file_name: {file_name}")
        user_folder = session['folder']
        folder_path = os.path.join(app.config["FOLDER_UPLOAD"], user_folder, course_name)
        activations_path = os.path.join(folder_path, app.config['ACTIVATIONS_FILE'])
        if os.path.exists(activations_path):
            with open(activations_path, 'r') as f:
                activations = json.load(f)
            
            activations[file_name] = not activations.get(file_name, False)
            
            with open(activations_path, 'w') as f:
                json.dump(activations, f)
                
            app.logger.info(f"Updated activations for {file_name}. New status: {activations[file_name]}")
            #Now that we have updated the activations, we need to run the chunking, embedding and creating final data in the background
            #Now going to run the chunking, embeding and creating final data in the background
            chunk_documents_given_course_name(os.path.join(app.config['FOLDER_UPLOAD'], user_folder, course_name))
            app.logger.info(f"Completed chop_course_content")
            embed_documents_given_course_name(os.path.join(app.config['FOLDER_UPLOAD'], user_folder, course_name))
            app.logger.info(f"Completed embed_course_content")       
            app.logger.info(f"Creating final data for course: COURSE NAME: {course_name}")
            create_final_data_given_course_name(os.path.join(app.config['FOLDER_UPLOAD'], user_folder, course_name))
            app.logger.info(f" Completed final data for course: COURSE NAME: {course_name}")

            return jsonify(success=True, status=activations[file_name])
        else:
            app.logger.warning(f"Activation file not found for course: {course_name}")
            return jsonify(success=False, error="Activation file not found!")
    except Exception as e:
        app.logger.error(f"Exception in toggle_activation: {str(e)}", exc_info=True)
        return jsonify(success=False, error=str(e))

@app.route('/course-contents/<course_name>', methods=['GET', 'POST'])
@login_required
def course_contents(course_name):
    form = UploadSyllabus()
    session['course_name'] = course_name
    user_folder = session['folder']
    # Part 1: Upload Syllabus PDF file and Save the text version: Check if the form was sucessfully validated:
    if form.validate_on_submit():
       app.logger.info(f"Form validated successfully: {form}")
       save_syllabus(form, course_name)
       app.logger.info(f"Syllabus saved successfully - Now Back in 'course-content' Route")
       #Now going to run the chunking, embeding and creating final data in the background
       chunk_documents_given_course_name(os.path.join(app.config['FOLDER_UPLOAD'], user_folder, course_name))
       app.logger.info(f"Completed chop_course_content")
       embed_documents_given_course_name(os.path.join(app.config['FOLDER_UPLOAD'], user_folder, course_name))
       app.logger.info(f"Completed embed_course_content")       
       app.logger.info(f"Creating final data for course: COURSE NAME: {course_name}")
       create_final_data_given_course_name(os.path.join(app.config['FOLDER_UPLOAD'], user_folder, course_name))
       app.logger.info(f" Completed final data for course: COURSE NAME: {course_name}")
    # Part 2: Load Course Content: 
    user_folder = session['folder']
    folder_path = os.path.join(app.config["FOLDER_UPLOAD"], user_folder, course_name)
    # Check if metadata file exists, if not create a blank one
    meta_file_path = os.path.join(folder_path, "course_meta.json")
    if not os.path.exists(meta_file_path):
        metadata = {
            'classname': '',
            'professor': '',
            'assistants': '',
            'classdescription': '',
            'assistant_name': ''
        }
        with open(meta_file_path, 'w') as json_file:
            json.dump(metadata, json_file)
    else:
        with open(meta_file_path, 'r') as json_file:
            metadata = json.load(json_file)
        load_course_metadata(metadata)



    contents = get_content_files(folder_path, course_name)
    file_info = detect_final_data_files(folder_path) # for 'textchunks.npy' and 'textchunks-originaltext.csv' if they exist
    activations = check_and_update_activations_file(folder_path, course_name)
    contents_info = check_processed_files(contents, folder_path)
    for info in contents_info:
        filename = info[0]
        info.append(activations.get(filename, False))
    # Part 3: Load Syllabus:
    ### TO DO: REPLACE THIS WITH ANOTHER FUNCTION THAT LOADS THE SYLLABUS IN THE NEW WAY
    syllabus_file_name = "Syllabus-" + course_name
    syllabus_folder_path = os.path.join(app.config["FOLDER_UPLOAD"], user_folder, course_name, 'Textchunks')
    # if os.listdir(syllabus_folder_path) does not exist, then create it
    if not os.path.exists(syllabus_folder_path):
        os.mkdir(syllabus_folder_path)
    syllabus_contents = os.listdir(syllabus_folder_path)
    # Check if there is a file that starts with course_name + "Syllabus" in contents
    syllabus = None
    for file in syllabus_contents:
        if file.startswith("Syllabus-" + course_name):
            file_path = os.path.join(syllabus_folder_path, file)
            syllabus = read_csv_preview(file_path)
            break  # Exit the loop as soon as the first matching file is found



    
    return render_template(
       'course_contents.html', 
       course_name=course_name,
       contents_info = contents_info, 
       contents=contents,
       syllabus=syllabus,
       name=session.get('name'), 
       form=form,
       file_info=file_info,
       folder=user_folder,
       metadata=metadata,  # add metadata to the template
       )

@app.route('/course-contents-rename/<course_name>', methods=['GET', 'POST'])
@login_required
def course_contents_rename(course_name):
    session['course_name'] = course_name
    user_folder = session['folder']
    # Part 2: Load Course Content: 
    user_folder = session['folder']
    folder_path = os.path.join(app.config["FOLDER_PREUPLOAD"], user_folder, course_name)
    contents = get_content_files(folder_path, course_name)
    app.logger.info(f"Contents: {contents}")
    RENAME_INSRUCTIONS = app.config['RENAME_INSRUCTIONS']
    UPLOAD_INSTRUCTIONS = app.config['UPLOAD_INSTRUCTIONS']
    return render_template(
       'course_contents_rename.html', 
       course_name=course_name,
       contents=contents,
       UPLOAD_INSTRUCTIONS=UPLOAD_INSTRUCTIONS,
       RENAME_INSRUCTIONS=RENAME_INSRUCTIONS,
       name=session.get('name')
       )

#Used to move the contents from the PreUpload to the Upload Folder
@app.route('/move-contents/<course_name>', methods=['GET'])
@login_required
def move_contents(course_name):
    user_folder = session.get('folder')
    # Check if the course_name is provided
    if not course_name:
        flash('Course name not provided.', 'error')
        return redirect(request.referrer)

    # Define source and destination paths
    source_path = os.path.join(app.config["FOLDER_PREUPLOAD"], user_folder, course_name)
    destination_path = os.path.join(app.config["FOLDER_UPLOAD"], user_folder, course_name)

    # Check if the source directory exists
    if not os.path.exists(source_path):
        flash('Source directory does not exist.', 'error')
        return redirect(request.referrer)

    # Make sure the destination directory exists
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    try:
        # Move each item in the source directory to the destination directory
        for item in os.listdir(source_path):
            s_item = os.path.join(source_path, item)
            d_item = os.path.join(destination_path, item)
            shutil.move(s_item, d_item)
            app.logger.info(f"Moved '{s_item}' to '{d_item}'")
        flash('Contents moved successfully.', 'success')
    except Exception as e:
        app.logger.error(f"An error occurred while moving content: {e}", exc_info=True)
        flash('An error occurred while moving contents.', 'error')

    return redirect(url_for('course_contents',course_name=course_name))

#Used to return the Final Course NPY and CSV file size in JSON output 
@app.route('/course-file-info/<course_name>', methods=['GET'])
@login_required
def course_file_info(course_name):
    user_folder = session['folder']
    folder_path = os.path.join(app.config["FOLDER_UPLOAD"], user_folder, course_name)
    file_info = detect_final_data_files(folder_path)
    return jsonify(file_info)


@app.route('/update-course-metadata', methods=['POST'])
@login_required
def update_course_metadata():
    # Extract the form data from the request
    folder_name = request.form.get('course_name', '')
    metadata = {
        'classname': request.form.get('name', ''),
        'professor': request.form.get('professor', ''),
        'assistants': request.form.get('assistants', ''),
        'assistant_name': request.form.get('assistant_name', ''),
        'classdescription': request.form.get('classdescription', '')        
    }   

    # Your logic to save/update the metadata goes here
    # This could involve saving the data to a database, 
    # or as in your previous example, writing the data to a JSON file.
    # I'll give a simple example using a fictitious database function:

    try:
        # Assume save_course_metadata is a function that saves the course metadata to your data storage
        save_course_metadata(folder_name, metadata)
        flash('Metadata updated successfully!', 'success')
    except Exception as e:
        # Handle any exceptions that might arise
        app.logger.error(f"Error updating metadata: {e}", exc_info=True)
        flash(f'Error updating metadata; Contact Support', 'danger')
    # Redirect to a relevant page after saving
    # For instance, redirecting back to the course management or course content page
    return redirect(request.referrer)

@app.route('/preview-chunks/<course_name>', methods=['GET'])
@login_required
def preview_chunks(course_name):
    user_folder = session['folder']
    folder_path = os.path.join(app.config["FOLDER_UPLOAD"], user_folder, course_name, 'Textchunks')

    # Filter only csv files in the course content folder
    csv_files = [f for f in os.listdir(folder_path) 
        if os.path.isfile(os.path.join(folder_path, f)) 
        and not f.startswith('.') 
        and f.endswith('.csv')]
    
    # Read the second element of the second line of each csv file
    second_entries = []
    for csv_file in csv_files:
        try:
            with open(os.path.join(folder_path, csv_file), 'r') as f:
                csv_reader = reader(f)
                next(csv_reader, None)  # Skip the header
                second_entry = next(csv_reader, None)  # Get the second row
                if second_entry and len(second_entry) > 1:  # Ensure there's a second element
                    second_entry = second_entry[1]  # Get the second element
                second_entries.append(second_entry)
                #print(f"Second row, second element of {csv_file}: {second_entry}")  # Print second row for debugging
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")  # Print any errors encountered
    #print(f"CSV files: {csv_files}\nSecond entries: {second_entries}")
    return render_template('preview_chunks.html', course_name=course_name, zip=zip, csv_files=csv_files, second_entries=second_entries, name=session.get('name'))


@app.route('/preview-chunks-js/<course_name>/<content_name>', methods=['GET'])
@login_required
def preview_chunks_js(course_name, content_name):
    user_folder = session['folder']
    folder_path = os.path.join(app.config["FOLDER_UPLOAD"], user_folder, course_name, 'Textchunks')
    file_path = os.path.join(folder_path, content_name)
    # Assuming we are reading a CSV with preview content
    preview_data = read_csv_preview(file_path)  # This function should be implemented to read the CSV preview data
    
    return jsonify({'preview_content': preview_data})

@app.route('/course-syllabus/<course_name>', methods=['GET', 'POST'])
@login_required
def course_syllabus(course_name):
    # check if we have the Syllabus already for this course
    user_folder = session['folder']
    syllabus_folder_path = os.path.join(app.config["FOLDER_UPLOAD"], user_folder, course_name, 'Textchunks')
    syllabus_contents = os.listdir(syllabus_folder_path)
    # Check if there is a file that starts with course_name + "Syllabus" in contents
    syllabus = None
    for file in syllabus_contents:
        if file.startswith("Syllabus-" + course_name):
            file_path = os.path.join(syllabus_folder_path, file)
            syllabus = read_csv_preview(file_path)
            break  # Exit the loop as soon as the first matching file is found
    # we have now processed PDF Uploads, Syllabus Loading, Course Content Loading.
    return render_template(
       'course_syllabus.html', 
       course_name=course_name, 
       syllabus=syllabus,
       name=session.get('name'), folder=session.get('folder'), admin=session.get('admin')
       )

#This is used by the Flask_Dropone component
@app.route('/upload-file', methods=['GET', 'POST'])
@login_required
def upload_file():
    app.logger.info(f"Entered upload_file")
    print("Entered upload_file")
    user_folder = session.get('folder', None)
    course_name = request.form.get('course_name', None)

    if user_folder is None:
        flash("User folder not found. Please ensure you're logged in properly.", 'error')
        app.logger.error(f"User folder not found. Please ensure you're logged in properly. Contact Administrator for support!")
        return redirect(url_for('courses'))  

    if course_name is None:
        app.logger.error(f"Course name not provided. Contact Administrator for support!")
        flash("Course name not provided. Please select a valid course.", 'error')
        return redirect(url_for('courses'))

    try:
        if request.method == 'POST':
            file = request.files.get('file')
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                #if os.path.join(app.config['FOLDER_PREUPLOAD'], user_folder, course_name) does not exist, then create it
                if not os.path.exists(os.path.join(app.config['FOLDER_PREUPLOAD'], user_folder, course_name)):
                    app.logger.info(f"Creating folder: {os.path.join(app.config['FOLDER_PREUPLOAD'], user_folder, course_name)}")
                    if not os.path.exists(os.path.join(app.config['FOLDER_PREUPLOAD'], user_folder)):
                        if not os.path.exists(os.path.join(app.config['FOLDER_PREUPLOAD'])):
                            os.mkdir(os.path.join(app.config['FOLDER_PREUPLOAD']))
                            app.logger.info(f"Folder created successfully: {os.path.join(app.config['FOLDER_PREUPLOAD'])}")
                        os.mkdir(os.path.join(app.config['FOLDER_PREUPLOAD'], user_folder))
                        app.logger.info(f"Folder created successfully: {os.path.join(app.config['FOLDER_PREUPLOAD'], user_folder)}")
                    os.mkdir(os.path.join(app.config['FOLDER_PREUPLOAD'], user_folder, course_name))
                    app.logger.info(f"Folder created successfully: {os.path.join(app.config['FOLDER_PREUPLOAD'], user_folder, course_name)}")
                file.save(os.path.join(app.config['FOLDER_PREUPLOAD'], user_folder, course_name, filename))
                app.logger.info(f"File {filename} uploaded successfully - For course: '{course_name}' for user: '{user_folder}'")
                flash(f"File {filename} uploaded successfully.", 'success')
                return f"File {filename} uploaded successfully."
            else:
                app.logger.error(f"File {file.filename} not uploaded - For course: '{course_name}' for user: '{user_folder}'; Reason: Unsupported file type.")
                flash(f"File {file.filename} not uploaded. Unsupported file type.", 'error')  
                return "Unsupported file type.", 400

        return render_template('course_contents_rename.html', course_name=course_name, name=session.get('name'))

    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return "An error occurred.", 500

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    settings_data = read_from_file_json(SETTINGS_PATH) or {}
    if request.method == 'POST':
        for setting, details in settings_data.items():
            # Check if setting was submitted in the form and set to "True" if it was, otherwise "False"
            settings_data[setting]["Value"] = "True" if request.form.get(setting) else "False"
            app.logger.info(f"Setting value for {setting}: {settings_data[setting]['Value']}")
        # Save the updated settings to the JSON file
        if write_to_file_json(SETTINGS_PATH, settings_data):
            app.logger.info("Successfully wrote settings to file.")
        else:
            app.logger.error("Failed to write settings to file.")  
        # Update Flask's app.config with the new settings
        for key, value in settings_data.items():
            app.config[key.upper()] = True if value["Value"] == "True" else False
        flash('Settings have been updated!', 'success') 
    return render_template('settings.html', settings=settings_data, name=session.get('name'), folder=session.get('folder'), admin=session.get('admin'))


@app.route('/logs',methods=['GET', 'POST'])
@login_required
def view_logs():
    load_previous = request.args.get('previous', None)
    
    raw_lines = []
    #first we load the first log file
    with open('logs/alldayta.xyz.log', 'r') as f:
        raw_lines += f.readlines() # = f.readlines()
    
    #if load_previous: then load the previous log file
    if load_previous:
        #if load_previos is a number, then load all the log files from the last n days where n = load_previous
        if load_previous.isdigit():
            load_previous = int(load_previous)
            log_files = []
            for i in range(load_previous):
                #ignore for i = 0
                if i > 0:
                    log_files.append(f"logs/alldayta.xyz.log.{i}")
        #Now read the log files and add them to the raw_lines
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    raw_lines += f.readlines()
            except Exception as e:
                app.logger.error(f"Error reading log file {log_file}: {e}")





    # Consolidate log entries
    consolidated_logs = []
    current_log = ''
    for line in raw_lines:
        if re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}", line):
            if current_log:
                consolidated_logs.append(current_log)
            current_log = line
        else:
            current_log += line
    if current_log:
        consolidated_logs.append(current_log)


 
    def safe_search(pattern, log, flags=0, default="N/A"):
        """Helper function to safely perform regex search."""
        match = re.search(pattern, log, flags)
        if match:
            return match.groups()  # Return all groups as a tuple
        elif isinstance(default, tuple):
            return default
        else:
            return (default,)

    # Parsing the logs
    parsed_logs = []
    for log in consolidated_logs:
        timestamp = safe_search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})", log)
        #level = safe_search(r" (\w+):", log)
        #   message = safe_search(r": (.+?) \[in", log)
        #message = safe_search(r":(.*?)\s\[in", log, flags=re.DOTALL) 
        level = safe_search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} (\w+):", log)
        message = safe_search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \w+: (.+?) \[in", log, flags=re.DOTALL)
        file_path = safe_search(r"\[in (.*?)]", log)
        request_data = safe_search(r"Request: '(.*?)' \[(\w+)]", log, default=("N/A", "N/A"))
        #session_str = safe_search(r"Session: (.*?);", log)
        session_str_tuple = safe_search(r"Session: (.*?);", log)
        session_str = session_str_tuple[0] if session_str_tuple else ''
        print(session_str)
        # Convert the session string into a dictionary
        try:
            session_dict = ast.literal_eval(session_str)
        except (ValueError, SyntaxError):
            session_dict = {}

        user_agent = safe_search(r"User Agent: (.*)$", log)
        
        parsed_logs.append({
            'timestamp': timestamp,
            'level': level,
            'message': format_json(message[0]),
            'file_path': file_path,
            'request_url': request_data[0],
            'request_method': request_data[1],
            'session': session_dict,
            'user_agent': user_agent,
            'name': session_dict.get('name', 'N/A'),
            'folder': session_dict.get('folder', 'N/A'),
            'course_name': session_dict.get('course_name', 'N/A')
        })

    # Sorting logs based on timestamp in descending order.
    parsed_logs = sorted(parsed_logs, key=lambda x: x['timestamp'][0], reverse=True)    

    return render_template('logs.html', logs=parsed_logs, previous=load_previous)


#  --------------------Routes for Content Processing --------------------

@app.route('/chop-course-content', methods=['GET'])
@login_required
def chop_course_content():
    app.logger.info(f"Entered chop_course_content")
    course_name = request.args.get('course_name', None) 
    user_folder = session['folder']
    chunk_documents_given_course_name(os.path.join(app.config['FOLDER_UPLOAD'], user_folder, course_name))
    app.logger.info(f"Completed chop_course_content")
    return redirect(request.referrer)

@app.route('/embed-course-content', methods=['GET'])
@login_required
def embed_course_content():
    app.logger.info(f"Entered embed_course_content")
    course_name = request.args.get('course_name', None) 
    user_folder = session['folder']
    embed_documents_given_course_name(os.path.join(app.config['FOLDER_UPLOAD'], user_folder, course_name))
    app.logger.info(f"Completed embed_course_content")
    return redirect(request.referrer)

@app.route('/create-final-data-course-content', methods=['GET'])
@login_required
def create_final_data_course_content():
    app.logger.info(f"Entered create_final_data_course_content")    
    try:
        user_folder = session['folder']
        app.logger.info(f"Entering final data for course. - In the try block")
        course_name = request.args.get('course_name', None) 
        app.logger.info(f"Creating final data for course: COURSE NAME: {course_name}")
        create_final_data_given_course_name(os.path.join(app.config['FOLDER_UPLOAD'], user_folder, course_name))
        app.logger.info(f" Completed final data for course: COURSE NAME: {course_name}")
        return redirect(request.referrer)
    except Exception as e:
        app.logger.error(f"An error occurred: {e}")
        return redirect(request.referrer)

#  --------------------Routes for Chatting --------------------

# This is the route for the Chatbot

# Apply the retry decorator to the original function
#@retry_with_exponential_backoff
@app.route('/teaching-assistant', methods=['GET', 'POST'])
def teaching_assistant():
    global df_chunks, embedding
    course_name = request.args.get('course_name', None)
    folder = request.args.get('folder', None) 
    
    # setting a new session variable to save course name
    # we save this to the session so that we can use it in the logging
    session['course_name'] = course_name
    session['folder'] = folder

    app.logger.info(f"Entered teaching_assistant for course {course_name} and folder {folder}")    
    course_folder = os.path.join(app.config['FOLDER_UPLOAD'], folder, course_name)
    dataname = os.path.join(course_folder,"Textchunks")
    user_folder = session['folder']
    folder_path = os.path.join(app.config["FOLDER_UPLOAD"], user_folder, course_name)
    # Check if metadata file exists, if not create a blank one
    meta_file_path = os.path.join(folder_path, "course_meta.json")
    if not os.path.exists(meta_file_path):
        metadata = {
            'classname': '',
            'professor': '',
            'assistants': '',
            'classdescription': '',
            'assistant_name': ''
        }
        with open(meta_file_path, 'w') as json_file:
            json.dump(metadata, json_file)
    else:
        with open(meta_file_path, 'r') as json_file:
            metadata = json.load(json_file)
        load_course_metadata(metadata)



    classname = course_name
    professor = session['professor']
    assistants = session['assistants']
    classdescription = session['classdescription']
    assistant_name = session['assistant_name']
    instruct = ('I am an experimental virtual TA for your course in entrepreneurship.'
                  ' I have been trained with all of your readings, course materials, lecture content, and slides. '
                  ' I am generally truthful, but be aware that there is a large language model in the background and hallucinations are possible. '
                  ' The more precise your question, the better an answer you will get. '
                  ' You may ask me questions in the language of your choice.  '
                  ' If "an error occurs while processing", ask your question again: '
                  ' the servers we use to process these answers are also in beta.')
    num_chunks = 8

    # check if we have the Syllabus already for this course
    if request.method == 'POST':
        with load_lock:
            # Load the text and its embeddings
            print("ok, starting")
            app.logger.info(f"Entered teaching_assistant for course {course_name} and folder {folder} - in the POST block")
            start_time = time.time()  # record the start time
            # add the log entry of the start time
            app.logger.info(f"Start time: {start_time}")
            df_chunks = load_df_chunks(dataname) # get df_chunks from the global
            elapsed_time = time.time() - start_time  # calculate the elapsed time
            print(f"Data loaded. Time taken: {elapsed_time:.2f} seconds")
            app.logger.info(f"Data loaded. Time taken: {elapsed_time:.2f} seconds")
            original_question = request.form['content1']

            # if there is a previous question and it's not multiple choice or its answer, check to see if the new one is a syllabus q or followup
            # this works OK for now, it will work better with GPT4
            if not (request.form['content1'].startswith('m:') or request.form['content1'].startswith('M:') or request.form['content1'].startswith('a:')):
                # first let's see if it's on the syllabus
                send_to_gpt = []
                send_to_gpt.append({
                    "role": "user",
                    "content": (
                        f"This question is from a student in an {classname} taught by "
                        f"{professor} with the help of {assistants}. The class is "
                        f"{classdescription} I want to know whether this question is "
                        f"likely about the logistical details, schedule, nature, teachers, "
                        f"assignments, or syllabus of the course? Answer Yes or No and "
                        f"nothing else: {request.form['content1']}"
                    )
                })
                #Log send_to_gpt variable
                formatted_send_to_gpt = json.dumps(str(send_to_gpt), indent=4, ensure_ascii=False)
                app.logger.info(f"Check if this is Syllabus question: 'Send_to_gpt' variable: {formatted_send_to_gpt}")
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    max_tokens=1,
                    temperature=0.0,
                    messages=send_to_gpt
                )
                print("Is this a syllabus question? " + response["choices"][0]["message"]["content"])
                app.logger.info(f"Is this a syllabus question? " + response["choices"][0]["message"]["content"])
                #Logging the whole response JSON
                # Convert the response object to a JSON-formatted string
                formatted_response = json.dumps(str(response), indent=4, ensure_ascii=False)
                # Log the formatted string
                app.logger.info(f"This is the fortmatted 'response' JSON:\n{formatted_response}")
                #app.logger.info(f"This is the 'response' JSON:" + str(response))

                # Construct new prompt if AI says that this is a syllabus question
                if response["choices"][0]["message"]["content"].startswith('Y') or response["choices"][0]["message"][
                    "content"].startswith('y'):
                    # Concatenate the strings to form the original_question value
                    print("It seems like this question is about the syllabus")
                    app.logger.info(f"It seems like this question is about the syllabus")
                    original_question = "I may be asking about a detail on the syllabus for " + classname + ". " + request.form['content1']
                    app.logger.info(f"Original question: {original_question}")
                # This follow-up question works great in 4 BUT NOT WELL WITH 3.5
                else:
                    # if not on the syllabus, and it might be a followup, see if it is
                    if len(request.form['content2'])>1:
                        send_to_gpt = []
                        content_value = (
                            f"Consider this new question from a user: {request.form['content1']}. "
                            f"Their prior question and the response was {request.form['content2']} "
                            f"Would it be helpful to have the context of the previous question and response to answer the new one? "
                            "For example, the new question may refer to 'this' or 'that' or 'the company' or 'their' or 'his' "
                            "or 'her' or 'the paper' or similar terms whose context is not clear if you only know the current question "
                            "and don't see the previous question and response, or it may ask for more details or to summarize or "
                            "rewrite or expand on the prior answer in a way that is impossible to do unless you can see the previous answer. "
                            "Answer either Yes or No."
                        )
                        send_to_gpt.append({"role": "user",
                                            "content": content_value})
                        response = openai.ChatCompletion.create(
                            model="gpt-4",
                            max_tokens=1,
                            temperature=0.0,
                            messages=send_to_gpt
                        )
                        print(f"Consider this new question from a user: {request.form['content1']}."
                            f" Their prior question and the response was {request.form['content2']}."
                            " Think very logically."
                            " Is the information requested in the new question potentially related to the previous question and response?"
                            " Answer either Yes or No.")
                        print("Might this be a follow-up? " + response["choices"][0]["message"]["content"])
                        # Construct new prompt if AI says that this is a followup
                        if response["choices"][0]["message"]["content"].startswith('Y') or response["choices"][0]["message"]["content"].startswith('y'):
                           # Concatenate the strings to form the original_question value
                            print("Creating follow-up question")
                            #original_question = 'I have a followup on the previous question and response. ' + request.form['content2'] + 'My new question is: ' + request.form['content1']
                            original_question = ('I have a followup on the previous question and response. ' 
                                                + request.form['content2'] 
                                                + 'My new question is: ' 
                                                + request.form['content1'])







            # if answer to Q&A, don't embed a new search, just use existing context
            if request.form['content1'].startswith('a:'):
                print("Let's try to answer that question")
                most_similar = grab_last_response()
                title_str = "<p></p>"
                print("Query being used: " + request.form['content1'])
                print("The content we draw on begins " + most_similar[:200])
                elapsed_time = time.time() - start_time  # calculate the elapsed time
                print(f"Original context for question loaded. Time taken: {elapsed_time:.2f} seconds")
            else:
                embedthequery = openai.Embedding.create(
                    model="text-embedding-ada-002",
                    input=original_question
                )
                print("Query we asked is: " + original_question)
                query_embed=embedthequery["data"][0]["embedding"]
                elapsed_time = time.time() - start_time  # calculate the elapsed time
                print(f"Query embedded. Time taken: {elapsed_time:.2f} seconds")

                # function to compute dot product similarity; tested using Faiss library and didn't really help
                def compute_similarity(embedding, userquery):
                   similarities = np.dot(embedding, userquery)
                   return similarities
                # compute similarity for each row and add to new column
                print(len(query_embed))
                df_chunks['similarity'] = np.dot(embedding, query_embed)
                # sort by similarity in descending order
                df_chunks = df_chunks.sort_values(by='similarity', ascending=False)
                # Select the top query_similar_number most similar articles
                most_similar_df = df_chunks.head(num_chunks)
                elapsed_time = time.time() - start_time  # calculate the elapsed time
                print(f"Original query similarity sorted. Time taken: {elapsed_time:.2f} seconds")
                # Drop duplicate rows based on Title and Text columns
                most_similar_df = most_similar_df.drop_duplicates(subset=['Title', 'Text'])
                # Count the number of occurrences of each title in most_similar_df
                title_counts = most_similar_df['Title'].value_counts()
                # Create a new dataframe with title and count columns, sorted by count in descending order
                title_df = pd.DataFrame({'Title': title_counts.index, 'Count': title_counts.values}).sort_values('Count', ascending=False)
                # Filter the titles that appear at least three times
                title_df_filtered = title_df[title_df['Count'] >= 3]
                # Get the most common titles in title_df_filtered
                titles = title_df_filtered['Title'].values.tolist()
                if len(titles) == 1:
                    title_str = f'<span style="float:right;" id="moreinfo"><a href="#" onclick="toggle_visibility(\'sorting\');" style="text-decoration: none; color: black;">&#9776;</a><div id="sorting" style="display:none; font-size: 12px;"> [The most likely related text is "{titles[0]}"]</div></span><p>'
                    title_str_2 = f'The most likely related text is {titles[0]}. '
                elif len(titles) == 0:
                    title_str = "<p></p>"
                    title_str_2 = ""
                else:
                    top_two_titles = titles[:2]
                    title_str = f'<span style="float:right;" id="moreinfo"><a href="#" onclick="toggle_visibility(\'sorting\');" style="text-decoration: none; color: black;">&#9776;</a><div id="sorting" style="display:none; font-size: 12px;"> [The most likely related texts are "{top_two_titles[0]}" and "{top_two_titles[1]}"]</div></span><p>'
                    title_str_2 = f'The most likely related texts are {top_two_titles[0]} and {top_two_titles[1]}. '
                elapsed_time = time.time() - start_time  # calculate the elapsed time
                print(f"Most related texts are {titles[:1]}.")
                most_similar = '\n\n'.join(row[1] for row in most_similar_df.values)








            # I use very low temperature to give most "factual" answer
            if request.form['content1'].startswith('m:'):
                instructions = "You are a very truthful, precise TA in a " + classname + ".  You think step by step. A strong graduate student is using you as a tutor.  The student would like you to prepare a challenging multiple choice question on the requested topic drawing ONLY on the attached context.  You do not have to merely ask about definitions, but can also construct scenarios or creative examples. NEVER refer to 'the attached context' or 'according to the article' or similar. Assume the student has no idea what context you are drawing your question from, and NEVER state the context you are drawing the question from: just state the question, then state options A to D. After the question, write <span style=\"display:none\"> then give your answer and a short explanation, then after your answer and explanation close the span with </span>"
                original_question = "Construct a challenging multiple-choice question to test me on a concept related to " + request.form['content1'][len('m:'):].strip()
                temperature=0.2
                # save question content for response
                truncated_most_similar = most_similar[:3900]
                session['last_session'] = truncated_most_similar
                print("saving old context to session variable")
            elif request.form['content1'].startswith('a:'):
                #instructions = "You are a very truthful, precise TA in a " + classname + ".  You think step by step. You are testing a strong graduate student on their knowledge.  The student would like you, using the attached context, to tell them whether they have answered the attached multiple choice question correctly.  Draw ONLY on the attached context for definitions and theoretical content.  Never refer to 'the attached context' or 'the article says that' or other context: just state your answer and the rationale."
                instructions = ("You are a very truthful, precise TA in a " + classname + ". "
               "You think step by step. You are testing a strong graduate student on their knowledge. "
               "The student would like you, using the attached context, to tell them whether they have "
               "answered the attached multiple choice question correctly. "
               "Draw ONLY on the attached context for definitions and theoretical content. "
               "Never refer to 'the attached context' or 'the article says that' or other context: "
               "just state your answer and the rationale.")

                original_question =  request.form['content1'][len('a:'):].strip()
                temperature=0.2
            else:
                instructions = ("You are a very truthful, precise TA in a " 
                                + classname + ", a " + classdescription + 
                                ".  You think step by step. A strong graduate student is asking you questions."  
                                " The answer to their query may appear in the attached book chapters, handouts, transcripts, and articles."  
                                " If it does, in no more than three paragraphs answer the user's question;" 
                                " you may answer in longer form with more depth if you need it to fully construct a requested numerical example."  
                                " Do not restate the question, do not refer to the context where you learned the answer, do not say you are an AI;" 
                                " just answer the question.  Say 'I don't know' if you can't find the answer to the original question in the text below;" 
                                " be very careful to match the terminology and definitions, implicit or explicit, used in the attached context." 
                                " You may try to derive more creative examples ONLY if the user asks for a numerical example of some type when" 
                                " you can construct it precisely using the terminology found in the attached context with high certainty," 
                                " or when you are asked for an empirical example or an application of an idea to a new context," 
                                " and you can construct one using the exact terminology and definitions in the text; remember," 
                                " you are a precise TA who wants the student to understand but also wants to make sure you do not" 
                                " contradict the readings and lectures the student has been given in class." 
                                " Please answer in the language of the student's question."
                                )
                temperature=0.2
            reply = []
            print("The question sent to GPT is " + original_question)
            print("The related content is: " + most_similar[:1000])
            send_to_gpt = []
            send_to_gpt.append({"role":"system","content":instructions + most_similar})
            send_to_gpt.append({"role":"user","content":original_question})
            response=openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=send_to_gpt
            )
            query = request.form['content1']
            tokens_new = response["usage"]["total_tokens"]
            reply1 = response["choices"][0]["message"]["content"]
            elapsed_time = time.time() - start_time  # calculate the elapsed time
            send_to_gpt = []
            print(f"GPT Response gathered. You used {tokens_new} tokens. Time taken: {elapsed_time:.2f} seconds")
            # check to make sure GPT is happy with the answer
            send_to_gpt.append({"role":"system","content":"Just say 'Yes' or 'No'. Do not give any other answer."})
            send_to_gpt.append({"role":"user","content":f"User: {original_question}  Attendant: {reply1} Was the Attendant able to answer the user's question?"})
            response=openai.ChatCompletion.create(
                model="gpt-4",
                max_tokens=1,
                temperature=0.0,
                messages=send_to_gpt
            )
            print("Did we answer the question? " + response["choices"][0]["message"]["content"])
            # if you don't find the answer, grab the most article we think is most related and try to prompt a followup





            if response["choices"][0]["message"]["content"].lower().startswith("no") and not request.form['content1'].startswith('a:'):
                send_to_gpt = []
                # need to reload df_chunks since its order no longer syncs with embeddings
                df_chunks = pd.read_csv(dataname + "-originaltext.csv")
                # get the article that was most related when we couldn't find the answer
                mostcommontitle = title_df["Title"].value_counts().index[0]
                title_counts = title_df["Title"].value_counts()




                # Check if there is more than one entry in title_df and the second most common title appears at least twice
                if len(title_counts) > 1 and title_counts.iloc[1] >= 2:
                    secondmostcommontitle = title_counts.index[1]
                    # now prompt again, giving that article as context
                    followup_input = f'Using {mostcommontitle}: {original_question}'
                    embedthequery2 = openai.Embedding.create(
                        model="text-embedding-ada-002",
                        input=followup_input
                    )
                    query_embed2 = embedthequery2["data"][0]["embedding"]
                    followup_input2 = f'Using {secondmostcommontitle}: {original_question}'
                    embedthequery3 = openai.Embedding.create(
                        model="text-embedding-ada-002",
                        input=followup_input
                    )
                    query_embed3 = embedthequery3["data"][0]["embedding"]
                    # compute similarity for each row and add to new column
                    df_chunks['similarity2'] = np.dot(embedding, query_embed2)
                    df_chunks['similarity3'] = np.dot(embedding, query_embed3)
                    # sort by similarity in descending order
                    df_chunks = df_chunks.sort_values(by='similarity2', ascending=False)
                    # Select the top query_similar_number most similar articles
                    most_similar_df_fhead = df_chunks.head(5)
                    print(df_chunks.head(2))
                    # sort by similarity in descending order
                    df_chunks = df_chunks.sort_values(by='similarity3', ascending=False)
                    # Select the top query_similar_number most similar articles
                    most_similar_df_fhead = pd.concat([most_similar_df_fhead, df_chunks.head(5)], axis=1)
                    print(df_chunks.head(2))
                    elapsed_time = time.time() - start_time  # calculate the elapsed time
                    print(f"Followup queries similarity sorted. Time taken: {elapsed_time:.2f} seconds")
                    # Drop duplicate rows based on Title and Text columns
                    most_similar_df_follow = most_similar_df_fhead.drop_duplicates(subset=['Title', 'Text'])
                    mostcommontitle = mostcommontitle + " and " + secondmostcommontitle
                    most_similar_followup = "The best guess at related texts is/are " + mostcommontitle + '\n\n'.join(
                        row[1] for row in most_similar_df_follow.values)
                    



                else:
                    # now prompt again, giving that article as context
                    followup_input = f'Using {mostcommontitle}: {original_question}'
                    embedthequery2 = openai.Embedding.create(
                        model="text-embedding-ada-002",
                        input=followup_input
                    )
                    query_embed2 = embedthequery2["data"][0]["embedding"]
                    # compute similarity for each row and add to new column
                    df_chunks['similarity2'] = np.dot(embedding, query_embed2)
                    # sort by similarity in descending order
                    df_chunks = df_chunks.sort_values(by='similarity2', ascending=False)
                    # Select the top query_similar_number most similar articles
                    most_similar_df_fhead = df_chunks.head(10)
                    print(df_chunks.head(2))
                    elapsed_time = time.time() - start_time  # calculate the elapsed time
                    print(f"Followup query similarity sorted. Time taken: {elapsed_time:.2f} seconds")
                    # Drop duplicate rows based on Title and Text columns
                    most_similar_df_follow = most_similar_df_fhead.drop_duplicates(subset=['Title', 'Text'])
                    most_similar_followup = "The best guess at related texts is/are " + mostcommontitle + '\n\n'.join(
                        row[1] for row in most_similar_df_follow.values)
                # now prompt again, giving that article as context
                followup_input=f'Using {mostcommontitle}: {original_question}'
                send_to_gpt.append({"role": "system", "content": instructions + most_similar_followup})
                send_to_gpt.append({"role": "user", "content": f'Using {mostcommontitle}: {original_question}'})
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=send_to_gpt
                )
                tokens_new = tokens_new+response["usage"]["total_tokens"]
                reply1 = response["choices"][0]["message"]["content"]
                # check to see if follow-up was answered
                send_to_gpt = []
                send_to_gpt.append(
                    {"role": "system", "content": "Just say 'Yes' or 'No'. Do not give any other answer."})
                send_to_gpt.append({"role": "user",
                                    "content": f"User: {original_question}  Attendant: {reply1} Was the Attendant able to answer the user's question?"})
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    max_tokens=1,
                    temperature=0.0,
                    messages=send_to_gpt
                )

                print("Did we answer the followup question? " + response["choices"][0]["message"]["content"])

                if response["choices"][0]["message"]["content"].lower().startswith("no") and not request.form['content1'].startswith('a:'):
                    reply1="I'm sorry but I cannot answer that question.  Can you rephrase or ask an alternative?"
                tokens_new = tokens_new + response["usage"]["total_tokens"]
                
            reply1=reply1.replace('\n', '<p>')
            reply = reply1 + title_str
            print(tokens_new)
            return reply
        



    else:
        # Start background thread to load data
        thread = threading.Thread(target=background_loading, args=(dataname,))
        thread.start()
    


    ####------------- TO DO : LOAD SYLLABUS
    # This is now manually being triggered by the faculty. 
    #if courses_with_final_data(app.config['FOLDER_UPLOAD']):
    #  courses = courses_with_final_data(app.config['FOLDER_UPLOAD'])
    #else:
    #   syllabus = None
    # we have now processed PDF Uploads, Syllabus Loading, Course Content Loading.
    return render_template(
       'ta.html', 
       # courses=courses, # THis is no longer needed
       name=session.get('name'),
       course_name = course_name,
       instruct = ('I am an experimental virtual TA for your course in <i>' 
                   + course_name + 
                   '</i>.<br>I have been trained with all of your readings, course materials, lecture content, and slides.<br>'
                   'I am generally truthful, but be aware that there is a large language model in the background and hallucinations are possible. <br>'
                   'The more precise your question, the better an answer you will get. You may ask me questions in the language of your choice. <br>'
                    'If "an error occurs while processing", ask your question again: the servers we use to process these answers are also in beta.')
                    )




# Apply the retry decorator to the original function
#@retry_with_exponential_backoff
@app.route('/teaching-assistant-student', methods=['GET', 'POST'])
def teaching_assistant_student():
    global df_chunks, embedding
    course_name = request.args.get('course_name', None)
    folder = request.args.get('folder', None) 
    
    # setting a new session variable to save course name
    # we save this to the session so that we can use it in the logging
    session['course_name'] = course_name
    session['folder'] = folder

    app.logger.info(f"Entered teaching_assistant for course {course_name} and folder {folder}")    
    course_folder = os.path.join(app.config['FOLDER_UPLOAD'], folder, course_name)
    dataname = os.path.join(course_folder,"Textchunks")
    user_folder = session['folder']
    folder_path = os.path.join(app.config["FOLDER_UPLOAD"], user_folder, course_name)
    # Check if metadata file exists, if not create a blank one
    meta_file_path = os.path.join(folder_path, "course_meta.json")
    if not os.path.exists(meta_file_path):
        metadata = {
            'classname': '',
            'professor': '',
            'assistants': '',
            'classdescription': '',
            'assistant_name': ''
        }
        with open(meta_file_path, 'w') as json_file:
            json.dump(metadata, json_file)
    else:
        with open(meta_file_path, 'r') as json_file:
            metadata = json.load(json_file)
        load_course_metadata(metadata)



    classname = course_name
    professor = session['professor']
    assistants = session['assistants']
    classdescription = session['classdescription']
    assistant_name = session['assistant_name']
    instruct = ('I am an experimental virtual TA for your course in entrepreneurship.'
                  ' I have been trained with all of your readings, course materials, lecture content, and slides. '
                  ' I am generally truthful, but be aware that there is a large language model in the background and hallucinations are possible. '
                  ' The more precise your question, the better an answer you will get. '
                  ' You may ask me questions in the language of your choice.  '
                  ' If "an error occurs while processing", ask your question again: '
                  ' the servers we use to process these answers are also in beta.')
    num_chunks = 8
    # writing the professor, assistants, class description, and assistant name to the app logger
    app.logger.info(f"Professor: {professor}")
    app.logger.info(f"Assistants: {assistants}")
    app.logger.info(f"Class Description: {classdescription}")
    app.logger.info(f"Assistant Name: {assistant_name}")



    #write the request.method to the app logger
    app.logger.info(f"Request method: {request.method}")

    # check if we have the Syllabus already for this course
    if request.method == 'POST':
        with load_lock:
            # Load the text and its embeddings
            print("ok, starting")
            app.logger.info(f"Entered teaching_assistant for course {course_name} and folder {folder} - in the POST block")
            start_time = time.time()  # record the start time
            # add the log entry of the start time
            app.logger.info(f"Start time: {start_time}")
            df_chunks = load_df_chunks(dataname) # get df_chunks from the global
            elapsed_time = time.time() - start_time  # calculate the elapsed time
            print(f"Data loaded. Time taken: {elapsed_time:.2f} seconds")
            app.logger.info(f"Data loaded. Time taken: {elapsed_time:.2f} seconds")
            original_question = request.form['content1']

            # if there is a previous question and it's not multiple choice or its answer, check to see if the new one is a syllabus q or followup
            # this works OK for now, it will work better with GPT4
            if not (request.form['content1'].startswith('m:') or request.form['content1'].startswith('M:') or request.form['content1'].startswith('a:')):
                # first let's see if it's on the syllabus
                send_to_gpt = []
                send_to_gpt.append({
                    "role": "user",
                    "content": (
                        f"This question is from a student in an {classname} taught by "
                        f"{professor} with the help of {assistants}. The class is "
                        f"{classdescription} I want to know whether this question is "
                        f"likely about the logistical details, schedule, nature, teachers, "
                        f"assignments, or syllabus of the course? Answer Yes or No and "
                        f"nothing else: {request.form['content1']}"
                    )
                })
                #Log send_to_gpt variable
                formatted_send_to_gpt = json.dumps(str(send_to_gpt), indent=4, ensure_ascii=False)
                app.logger.info(f"Check if this is Syllabus question: 'Send_to_gpt' variable: {formatted_send_to_gpt}")
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    max_tokens=1,
                    temperature=0.0,
                    messages=send_to_gpt
                )
                print("Is this a syllabus question? " + response["choices"][0]["message"]["content"])
                app.logger.info(f"Is this a syllabus question? " + response["choices"][0]["message"]["content"])
                #Logging the whole response JSON
                # Convert the response object to a JSON-formatted string
                formatted_response = json.dumps(str(response), indent=4, ensure_ascii=False)
                # Log the formatted string
                app.logger.info(f"This is the fortmatted 'response' JSON:\n{formatted_response}")
                #app.logger.info(f"This is the 'response' JSON:" + str(response))

                # Construct new prompt if AI says that this is a syllabus question
                if response["choices"][0]["message"]["content"].startswith('Y') or response["choices"][0]["message"][
                    "content"].startswith('y'):
                    # Concatenate the strings to form the original_question value
                    print("It seems like this question is about the syllabus")
                    app.logger.info(f"It seems like this question is about the syllabus")
                    original_question = "I may be asking about a detail on the syllabus for " + classname + ". " + request.form['content1']
                    app.logger.info(f"Original question: {original_question}")
                # This follow-up question works great in 4 BUT NOT WELL WITH 3.5
                else:
                    # if not on the syllabus, and it might be a followup, see if it is
                    if len(request.form['content2'])>1:
                        send_to_gpt = []
                        content_value = (
                            f"Consider this new question from a user: {request.form['content1']}. "
                            f"Their prior question and the response was {request.form['content2']} "
                            f"Would it be helpful to have the context of the previous question and response to answer the new one? "
                            "For example, the new question may refer to 'this' or 'that' or 'the company' or 'their' or 'his' "
                            "or 'her' or 'the paper' or similar terms whose context is not clear if you only know the current question "
                            "and don't see the previous question and response, or it may ask for more details or to summarize or "
                            "rewrite or expand on the prior answer in a way that is impossible to do unless you can see the previous answer. "
                            "Answer either Yes or No."
                        )
                        send_to_gpt.append({"role": "user",
                                            "content": content_value})
                        response = openai.ChatCompletion.create(
                            model="gpt-4",
                            max_tokens=1,
                            temperature=0.0,
                            messages=send_to_gpt
                        )
                        print(f"Consider this new question from a user: {request.form['content1']}."
                            f" Their prior question and the response was {request.form['content2']}."
                            " Think very logically."
                            " Is the information requested in the new question potentially related to the previous question and response?"
                            " Answer either Yes or No.")
                        print("Might this be a follow-up? " + response["choices"][0]["message"]["content"])
                        # Construct new prompt if AI says that this is a followup
                        if response["choices"][0]["message"]["content"].startswith('Y') or response["choices"][0]["message"]["content"].startswith('y'):
                           # Concatenate the strings to form the original_question value
                            print("Creating follow-up question")
                            #original_question = 'I have a followup on the previous question and response. ' + request.form['content2'] + 'My new question is: ' + request.form['content1']
                            original_question = ('I have a followup on the previous question and response. ' 
                                                + request.form['content2'] 
                                                + 'My new question is: ' 
                                                + request.form['content1'])







            # if answer to Q&A, don't embed a new search, just use existing context
            if request.form['content1'].startswith('a:'):
                print("Let's try to answer that question")
                most_similar = grab_last_response()
                title_str = "<p></p>"
                print("Query being used: " + request.form['content1'])
                print("The content we draw on begins " + most_similar[:200])
                elapsed_time = time.time() - start_time  # calculate the elapsed time
                print(f"Original context for question loaded. Time taken: {elapsed_time:.2f} seconds")
            else:
                embedthequery = openai.Embedding.create(
                    model="text-embedding-ada-002",
                    input=original_question
                )
                print("Query we asked is: " + original_question)
                query_embed=embedthequery["data"][0]["embedding"]
                elapsed_time = time.time() - start_time  # calculate the elapsed time
                print(f"Query embedded. Time taken: {elapsed_time:.2f} seconds")

                # function to compute dot product similarity; tested using Faiss library and didn't really help
                def compute_similarity(embedding, userquery):
                   similarities = np.dot(embedding, userquery)
                   return similarities
                # compute similarity for each row and add to new column
                print(len(query_embed))
                df_chunks['similarity'] = np.dot(embedding, query_embed)
                # sort by similarity in descending order
                df_chunks = df_chunks.sort_values(by='similarity', ascending=False)
                # Select the top query_similar_number most similar articles
                most_similar_df = df_chunks.head(num_chunks)
                elapsed_time = time.time() - start_time  # calculate the elapsed time
                print(f"Original query similarity sorted. Time taken: {elapsed_time:.2f} seconds")
                # Drop duplicate rows based on Title and Text columns
                most_similar_df = most_similar_df.drop_duplicates(subset=['Title', 'Text'])
                # Count the number of occurrences of each title in most_similar_df
                title_counts = most_similar_df['Title'].value_counts()
                # Create a new dataframe with title and count columns, sorted by count in descending order
                title_df = pd.DataFrame({'Title': title_counts.index, 'Count': title_counts.values}).sort_values('Count', ascending=False)
                # Filter the titles that appear at least three times
                title_df_filtered = title_df[title_df['Count'] >= 3]
                # Get the most common titles in title_df_filtered
                titles = title_df_filtered['Title'].values.tolist()
                if len(titles) == 1:
                    title_str = f'<span style="float:right;" id="moreinfo"><a href="#" onclick="toggle_visibility(\'sorting\');" style="text-decoration: none; color: black;">&#9776;</a><div id="sorting" style="display:none; font-size: 12px;"> [The most likely related text is "{titles[0]}"]</div></span><p>'
                    title_str_2 = f'The most likely related text is {titles[0]}. '
                elif len(titles) == 0:
                    title_str = "<p></p>"
                    title_str_2 = ""
                else:
                    top_two_titles = titles[:2]
                    title_str = f'<span style="float:right;" id="moreinfo"><a href="#" onclick="toggle_visibility(\'sorting\');" style="text-decoration: none; color: black;">&#9776;</a><div id="sorting" style="display:none; font-size: 12px;"> [The most likely related texts are "{top_two_titles[0]}" and "{top_two_titles[1]}"]</div></span><p>'
                    title_str_2 = f'The most likely related texts are {top_two_titles[0]} and {top_two_titles[1]}. '
                elapsed_time = time.time() - start_time  # calculate the elapsed time
                print(f"Most related texts are {titles[:1]}.")
                most_similar = '\n\n'.join(row[1] for row in most_similar_df.values)








            # I use very low temperature to give most "factual" answer
            if request.form['content1'].startswith('m:'):
                instructions = "You are a very truthful, precise TA in a " + classname + ".  You think step by step. A strong graduate student is using you as a tutor.  The student would like you to prepare a challenging multiple choice question on the requested topic drawing ONLY on the attached context.  You do not have to merely ask about definitions, but can also construct scenarios or creative examples. NEVER refer to 'the attached context' or 'according to the article' or similar. Assume the student has no idea what context you are drawing your question from, and NEVER state the context you are drawing the question from: just state the question, then state options A to D. After the question, write <span style=\"display:none\"> then give your answer and a short explanation, then after your answer and explanation close the span with </span>"
                original_question = "Construct a challenging multiple-choice question to test me on a concept related to " + request.form['content1'][len('m:'):].strip()
                temperature=0.2
                # save question content for response
                truncated_most_similar = most_similar[:3900]
                session['last_session'] = truncated_most_similar
                print("saving old context to session variable")
            elif request.form['content1'].startswith('a:'):
                #instructions = "You are a very truthful, precise TA in a " + classname + ".  You think step by step. You are testing a strong graduate student on their knowledge.  The student would like you, using the attached context, to tell them whether they have answered the attached multiple choice question correctly.  Draw ONLY on the attached context for definitions and theoretical content.  Never refer to 'the attached context' or 'the article says that' or other context: just state your answer and the rationale."
                instructions = ("You are a very truthful, precise TA in a " + classname + ". "
               "You think step by step. You are testing a strong graduate student on their knowledge. "
               "The student would like you, using the attached context, to tell them whether they have "
               "answered the attached multiple choice question correctly. "
               "Draw ONLY on the attached context for definitions and theoretical content. "
               "Never refer to 'the attached context' or 'the article says that' or other context: "
               "just state your answer and the rationale.")

                original_question =  request.form['content1'][len('a:'):].strip()
                temperature=0.2
            else:
                instructions = ("You are a very truthful, precise TA in a " 
                                + classname + ", a " + classdescription + 
                                ".  You think step by step. A strong graduate student is asking you questions."  
                                " The answer to their query may appear in the attached book chapters, handouts, transcripts, and articles."  
                                " If it does, in no more than three paragraphs answer the user's question;" 
                                " you may answer in longer form with more depth if you need it to fully construct a requested numerical example."  
                                " Do not restate the question, do not refer to the context where you learned the answer, do not say you are an AI;" 
                                " just answer the question.  Say 'I don't know' if you can't find the answer to the original question in the text below;" 
                                " be very careful to match the terminology and definitions, implicit or explicit, used in the attached context." 
                                " You may try to derive more creative examples ONLY if the user asks for a numerical example of some type when" 
                                " you can construct it precisely using the terminology found in the attached context with high certainty," 
                                " or when you are asked for an empirical example or an application of an idea to a new context," 
                                " and you can construct one using the exact terminology and definitions in the text; remember," 
                                " you are a precise TA who wants the student to understand but also wants to make sure you do not" 
                                " contradict the readings and lectures the student has been given in class." 
                                " Please answer in the language of the student's question."
                                )
                temperature=0.2
            reply = []
            print("The question sent to GPT is " + original_question)
            print("The related content is: " + most_similar[:1000])
            send_to_gpt = []
            send_to_gpt.append({"role":"system","content":instructions + most_similar})
            send_to_gpt.append({"role":"user","content":original_question})
            response=openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=send_to_gpt
            )
            query = request.form['content1']
            tokens_new = response["usage"]["total_tokens"]
            reply1 = response["choices"][0]["message"]["content"]
            elapsed_time = time.time() - start_time  # calculate the elapsed time
            send_to_gpt = []
            print(f"GPT Response gathered. You used {tokens_new} tokens. Time taken: {elapsed_time:.2f} seconds")
            # check to make sure GPT is happy with the answer
            send_to_gpt.append({"role":"system","content":"Just say 'Yes' or 'No'. Do not give any other answer."})
            send_to_gpt.append({"role":"user","content":f"User: {original_question}  Attendant: {reply1} Was the Attendant able to answer the user's question?"})
            response=openai.ChatCompletion.create(
                model="gpt-4",
                max_tokens=1,
                temperature=0.0,
                messages=send_to_gpt
            )
            print("Did we answer the question? " + response["choices"][0]["message"]["content"])
            # if you don't find the answer, grab the most article we think is most related and try to prompt a followup





            if response["choices"][0]["message"]["content"].lower().startswith("no") and not request.form['content1'].startswith('a:'):
                send_to_gpt = []
                # need to reload df_chunks since its order no longer syncs with embeddings
                df_chunks = pd.read_csv(dataname + "-originaltext.csv")
                # get the article that was most related when we couldn't find the answer
                mostcommontitle = title_df["Title"].value_counts().index[0]
                title_counts = title_df["Title"].value_counts()




                # Check if there is more than one entry in title_df and the second most common title appears at least twice
                if len(title_counts) > 1 and title_counts.iloc[1] >= 2:
                    secondmostcommontitle = title_counts.index[1]
                    # now prompt again, giving that article as context
                    followup_input = f'Using {mostcommontitle}: {original_question}'
                    embedthequery2 = openai.Embedding.create(
                        model="text-embedding-ada-002",
                        input=followup_input
                    )
                    query_embed2 = embedthequery2["data"][0]["embedding"]
                    followup_input2 = f'Using {secondmostcommontitle}: {original_question}'
                    embedthequery3 = openai.Embedding.create(
                        model="text-embedding-ada-002",
                        input=followup_input
                    )
                    query_embed3 = embedthequery3["data"][0]["embedding"]
                    # compute similarity for each row and add to new column
                    df_chunks['similarity2'] = np.dot(embedding, query_embed2)
                    df_chunks['similarity3'] = np.dot(embedding, query_embed3)
                    # sort by similarity in descending order
                    df_chunks = df_chunks.sort_values(by='similarity2', ascending=False)
                    # Select the top query_similar_number most similar articles
                    most_similar_df_fhead = df_chunks.head(5)
                    print(df_chunks.head(2))
                    # sort by similarity in descending order
                    df_chunks = df_chunks.sort_values(by='similarity3', ascending=False)
                    # Select the top query_similar_number most similar articles
                    most_similar_df_fhead = pd.concat([most_similar_df_fhead, df_chunks.head(5)], axis=1)
                    print(df_chunks.head(2))
                    elapsed_time = time.time() - start_time  # calculate the elapsed time
                    print(f"Followup queries similarity sorted. Time taken: {elapsed_time:.2f} seconds")
                    # Drop duplicate rows based on Title and Text columns
                    most_similar_df_follow = most_similar_df_fhead.drop_duplicates(subset=['Title', 'Text'])
                    mostcommontitle = mostcommontitle + " and " + secondmostcommontitle
                    most_similar_followup = "The best guess at related texts is/are " + mostcommontitle + '\n\n'.join(
                        row[1] for row in most_similar_df_follow.values)
                    



                else:
                    # now prompt again, giving that article as context
                    followup_input = f'Using {mostcommontitle}: {original_question}'
                    embedthequery2 = openai.Embedding.create(
                        model="text-embedding-ada-002",
                        input=followup_input
                    )
                    query_embed2 = embedthequery2["data"][0]["embedding"]
                    # compute similarity for each row and add to new column
                    df_chunks['similarity2'] = np.dot(embedding, query_embed2)
                    # sort by similarity in descending order
                    df_chunks = df_chunks.sort_values(by='similarity2', ascending=False)
                    # Select the top query_similar_number most similar articles
                    most_similar_df_fhead = df_chunks.head(10)
                    print(df_chunks.head(2))
                    elapsed_time = time.time() - start_time  # calculate the elapsed time
                    print(f"Followup query similarity sorted. Time taken: {elapsed_time:.2f} seconds")
                    # Drop duplicate rows based on Title and Text columns
                    most_similar_df_follow = most_similar_df_fhead.drop_duplicates(subset=['Title', 'Text'])
                    most_similar_followup = "The best guess at related texts is/are " + mostcommontitle + '\n\n'.join(
                        row[1] for row in most_similar_df_follow.values)
                # now prompt again, giving that article as context
                followup_input=f'Using {mostcommontitle}: {original_question}'
                send_to_gpt.append({"role": "system", "content": instructions + most_similar_followup})
                send_to_gpt.append({"role": "user", "content": f'Using {mostcommontitle}: {original_question}'})
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=send_to_gpt
                )
                tokens_new = tokens_new+response["usage"]["total_tokens"]
                reply1 = response["choices"][0]["message"]["content"]
                # check to see if follow-up was answered
                send_to_gpt = []
                send_to_gpt.append(
                    {"role": "system", "content": "Just say 'Yes' or 'No'. Do not give any other answer."})
                send_to_gpt.append({"role": "user",
                                    "content": f"User: {original_question}  Attendant: {reply1} Was the Attendant able to answer the user's question?"})
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    max_tokens=1,
                    temperature=0.0,
                    messages=send_to_gpt
                )

                print("Did we answer the followup question? " + response["choices"][0]["message"]["content"])

                if response["choices"][0]["message"]["content"].lower().startswith("no") and not request.form['content1'].startswith('a:'):
                    reply1="I'm sorry but I cannot answer that question.  Can you rephrase or ask an alternative?"
                tokens_new = tokens_new + response["usage"]["total_tokens"]
                
            reply1=reply1.replace('\n', '<p>')
            reply = reply1 + title_str
            print(tokens_new)
            return reply
        



    else:
        # Start background thread to load data
        thread = threading.Thread(target=background_loading, args=(dataname,))
        thread.start()
    


    ####------------- TO DO : LOAD SYLLABUS
    # This is now manually being triggered by the faculty. 
    #if courses_with_final_data(app.config['FOLDER_UPLOAD']):
    #  courses = courses_with_final_data(app.config['FOLDER_UPLOAD'])
    #else:
    #   syllabus = None
    # we have now processed PDF Uploads, Syllabus Loading, Course Content Loading.
    return render_template(
       'alldayta.html', 
       # courses=courses, # THis is no longer needed
       name=session.get('name'),
       course_name = course_name,
       assistants = assistant_name,
       folder=folder,
       instruct = ('I am an experimental virtual TA for your course in <i>' 
                   + course_name + 
                   '</i>.<br>I have been trained with all of your readings, course materials, lecture content, and slides.<br>'
                   'I am generally truthful, but be aware that there is a large language model in the background and hallucinations are possible. <br>'
                   'The more precise your question, the better an answer you will get. You may ask me questions in the language of your choice. <br>'
                    'If "an error occurs while processing", ask your question again: the servers we use to process these answers are also in beta.')
                    )


#  --------------------Functions for Chatting --------------------

# this ensures we load the data before taking an input
load_lock = threading.Lock()

def background_loading(dataname):
    with load_lock:
        global df_chunks, embedding
        df_chunks = load_df_chunks(dataname)
        print("Loaded data from background")

def grab_last_response():
    global last_session
    last_session = session.get('last_session', None)
    print("Ok, we have last session")
    if last_session is None:
        print("I don't know old content")
        last_session = ""
    return last_session

def load_df_chunks(dataname):
    global df_chunks, embedding
    # maybe just save this numpy embedding?
    if embedding is None:
        df_chunks = pd.read_csv(dataname+"-originaltext.csv")
        embedding = np.load(dataname+".npy")
        #print(f"embedding dimensions: {embedding.shape}")
        #print(f"df_chunks dimensions: {df_chunks.shape}")
    else:
        print("Database already loaded")
    return df_chunks

