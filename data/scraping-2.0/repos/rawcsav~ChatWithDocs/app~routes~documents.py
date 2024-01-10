import os
import threading

import openai
import pandas as pd
from flask import session, Blueprint, render_template, request, jsonify
from werkzeug.utils import secure_filename

from app.config import MAX_FILE_SIZE
from app.util import add_embeddings_to_df, gather_text_sections, extract_text_from_file, allowed_file

bp = Blueprint('documents', __name__)

lock = threading.Lock()


@bp.route('/')
def chat_index():
    os.makedirs(session['CHAT_UPLOAD_DIR'], exist_ok=True)

    uploaded_files = session.get('uploaded_files', [])
    existing_files = [f for f in uploaded_files if os.path.exists(os.path.join(session['CHAT_UPLOAD_DIR'], f))]

    session['uploaded_files'] = existing_files

    return render_template('index.html', uploaded_files=existing_files)


@bp.route('/upload', methods=['POST'])
def upload_file():
    api_key = session.get('api_key')
    openai.api_key = api_key
    response = {
        "status": "success",
        "messages": []
    }

    EMBED_DATA_PATH = session.get('EMBED_DATA')
    CHAT_UPLOAD_DIR = session.get('CHAT_UPLOAD_DIR')
    os.makedirs(CHAT_UPLOAD_DIR, exist_ok=True)

    if 'file' not in request.files:
        response["status"] = "error"
        response["messages"].append("No file part in the request.")
        return jsonify(response), 400

    uploaded_files = request.files.getlist('file')
    if not uploaded_files:
        response["status"] = "error"
        response["messages"].append("No file selected for uploading.")
        return jsonify(response), 400

    new_files = []

    for file in uploaded_files:
        secure_file_name = secure_filename(file.filename)
        if not file or file.content_length > MAX_FILE_SIZE or not allowed_file(secure_file_name):
            response["messages"].append(f"File {secure_file_name} is either too large or not of an allowed type.")
            continue
        if file and allowed_file(secure_file_name):
            filename = os.path.join(CHAT_UPLOAD_DIR, secure_file_name)
            file.save(filename)

            # Convert to .txt if it's a pdf or docx
            if filename.endswith('.pdf') or filename.endswith('.docx'):
                txt_filepath = extract_text_from_file(filename)
                if txt_filepath:
                    filename = txt_filepath
                else:
                    response["messages"].append(f"File {secure_file_name} conversion to .txt failed.")
                    continue

            new_files.append(os.path.basename(filename))
            response["messages"].append(f"File {os.path.basename(filename)} processed successfully.")

    try:
        df_new = gather_text_sections(CHAT_UPLOAD_DIR)
        df_new = add_embeddings_to_df(df_new, api_key)
    except Exception as e:
        response["status"] = "error"
        response["messages"].append(f"Error generating embeddings for {secure_file_name}: {str(e)}")

    with lock:
        if os.path.exists(EMBED_DATA_PATH) and os.path.getsize(EMBED_DATA_PATH) > 0:
            df_existing = pd.read_json(EMBED_DATA_PATH, orient='split')
        else:
            df_existing = pd.DataFrame()

        if df_existing.empty:
            df_combined = df_new
        else:
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)

    df_combined.to_json(EMBED_DATA_PATH, orient='split')

    uploaded_file_names = session.get('uploaded_files', [])

    session['uploaded_files'] = list(set(uploaded_file_names + new_files))

    return jsonify(response)
