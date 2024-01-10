import os
import re
import fitz
from io import BytesIO
from io import StringIO
import cv2
import numpy as np
from PIL import Image
import markdown
import openai
import pandas as pd
import pytesseract
from flask import Response
from flask import render_template, request, redirect, url_for, flash, session, send_from_directory, jsonify
from tabulate import tabulate
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from app import app, db
from app.models import User, UserFile


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Retrieve data from the form
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')

        # Hash the password
        hashed_password = generate_password_hash(password)

        # Create a new user instance
        new_user = User(
            first_name=first_name,
            last_name=last_name,
            email=email,
            username=username,
            password_hash=hashed_password
        )

        # Add the new user to the database
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful. Please log in.')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        login_input = request.form['email']  # This field will now accept either username or email
        password = request.form['password']

        # Determine if login_input is an email or username
        user = None
        if "@" in login_input:  # Assuming if input contains '@', it's an email
            user = User.query.filter_by(email=login_input).first()
        else:
            user = User.query.filter_by(username=login_input).first()

        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.user_id
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid login credentials')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))


@app.route('/')
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    user = User.query.get(user_id)
    user_files = UserFile.query.filter_by(user_id=user_id).all()

    if not user:
        flash('User not found.', 'error')
        return redirect(url_for('login'))

    # No changes are needed here if your User model has first_name and last_name fields
    return render_template('dashboard.html', user=user, user_files=user_files)

def preprocess_image_extract_text(file_path):
    """
    Apply advanced preprocessing steps to enhance image quality for OCR.
    """
    # Read the image using OpenCV
    img = cv2.imread(file_path)

    if img is None:
        return ""

    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Binarization using Otsu's method
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 5: Morphological operations to clear noise
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Use PIL to convert to image (if required by pytesseract version)
    final_image = Image.fromarray(opening)

    # Step 6: Tesseract OCR with custom configuration
    custom_config = r'--oem 3 --psm 6'
    extracted_text = pytesseract.image_to_string(final_image, config=custom_config)

    return extracted_text


@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])

    if request.method == 'POST':
        file = request.files.get('file')

        if not file:
            flash('No file part', 'error')
            return redirect(request.url)

        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash('Invalid file type', 'error')
            return redirect(request.url)

        user_folder = os.path.join(app.config['UPLOAD_FOLDER'], user.username)
        os.makedirs(user_folder, exist_ok=True)

        filename = secure_filename(file.filename)
        file_path = os.path.join(user_folder, filename)

        file.save(file_path)

        # Check the file extension to determine the processing method
        file_extension = os.path.splitext(filename)[1].lower()

        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']

        # Initialize extracted_text
        extracted_text = ""
        if file_extension in image_extensions:
            try:
                extracted_text = preprocess_image_extract_text(file_path)
            except Exception as e:
                print(f'Error processing file: {e}')

        elif file_extension in '.pdf':
            with fitz.open(file_path) as pdf:
                for page in pdf:
                    extracted_text += page.get_text()

        print(extracted_text)
        # Check if the file already exists
        existing_file = UserFile.query.filter_by(user_id=user.user_id, file_name=filename).first()
        if existing_file:
            os.remove(os.path.join(user_folder, existing_file.file_name))
            existing_file.file_path = file_path
            existing_file.file_content = extracted_text  # Update the existing record
        else:
            # Create a new record for the new file
            new_file = UserFile(user_id=user.user_id, file_name=filename, file_path=file_path,
                                file_content=extracted_text)
            db.session.add(new_file)

        db.session.commit()
        return redirect(url_for('dashboard'))

    return render_template('dashboard.html')


@app.route('/delete_file/<int:file_id>')
def delete_file(file_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    file_to_delete = UserFile.query.get(file_id)
    if file_to_delete and file_to_delete.user_id == session['user_id']:
        user = User.query.get(session['user_id'])
        user_folder = os.path.join(app.config['UPLOAD_FOLDER'], user.username)
        file_path = os.path.join(user_folder, file_to_delete.file_name)

        if os.path.exists(file_path):
            os.remove(file_path)

        db.session.delete(file_to_delete)
        db.session.commit()
        # flash('File deleted successfully')
    else:
        flash('Unauthorized to delete this file')
    return redirect(url_for('dashboard'))


@app.route('/uploads/<username>/<filename>')
def uploaded_file(username, filename):
    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], username)
    return send_from_directory(user_folder, filename)


def shuffle_text_function(text):
    return reversed(text)


def convert_markdown_to_html(text):
    # Convert Markdown text to HTML
    html = markdown.markdown(text)
    return html


def convert_text_to_html(text):
    # Split the text into lines
    lines = text.split('\n')

    # Start building the HTML content
    html_content = '<div class="text-content">'

    for line in lines:
        if line.strip():  # Check if the line is not empty
            # Wrap each line in a paragraph
            html_content += f'<p>{line}</p>'
        else:
            # Add a separator for empty lines (optional, for better readability)
            html_content += '<hr>'

    html_content += '</div>'  # Close the main div

    return html_content


def markdown_table_to_dataframe(full_text):
    # Split the text into lines
    lines = full_text.strip().split('\n')

    # Identify the header line and the separator line
    if len(lines) < 3:
        raise ValueError("Markdown table must have a header, separator, and at least one data row.")

    header_line = lines[0].strip()
    separator_line = lines[1].strip()

    # Validate the separator line (it should consist of dashes and pipes)
    if not all(char in ['-', '|', ' '] for char in separator_line):
        raise ValueError("Invalid separator line in markdown table.")

    # Exclude the separator line and any empty lines
    valid_lines = [line for line in lines if line.strip() and line != separator_line]

    # Rejoin the valid lines into a single string
    table_str = '\n'.join(valid_lines)

    # Convert the string to a DataFrame using pandas
    try:
        df = pd.read_csv(StringIO(table_str), sep='|')
        # Remove columns that are entirely NaN (occurs due to extra pipes)
        df = df.dropna(axis=1, how='all')
        # Clean up the DataFrame
        df = (df.apply(lambda x: x.str.strip() if x.dtype == "object" else x).dropna(how='all', axis=1)
        .dropna(how='all',axis=0).reset_index( drop=True))

        # Remove columns that are named like 'Unnamed: X'
        df = df.loc[:, ~df.columns.str.match('Unnamed:')]
        return df
    except Exception as e:
        raise ValueError(f"Error converting markdown table to DataFrame: {e}")


def send_excel_file_in_response(excel_io, filename):
    return Response(
        excel_io.getvalue(),
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        headers={'Content-Disposition': f'attachment;filename={filename}'}
    )


@app.route('/answer', methods=['POST'])
def smart_answer():
    if 'user_id' not in session:
        return jsonify({'error': 'User not logged in'}), 401

    # Retrieve the last exchange from the session
    last_exchange = session.get('last_exchange', [])

    user_id = session['user_id']
    user_files = UserFile.query.filter_by(user_id=user_id).all()
    user_query = request.form.get('query')

    # Get the selected file ID and Excel download checkbox status
    selected_file_id = request.form.get('fileId')
    print("selected field")
    print(selected_file_id)
    if selected_file_id:
        try:
            selected_file_id = int(selected_file_id)
        except ValueError:
            selected_file_id = None

    excel_selected = request.form.get('excelSelected') == 'true'

    # Generate the context from the files
    context = format_file_contents(user_files, selected_file_id)
    print(context)

    # Generate response using the provided context and other parameters
    ai_response, updated_last_exchange = generate_response(user_query, context, last_exchange, excel_selected,
                                                           selected_file_id)

    # Update the session to store only the immediate last exchange
    session['last_exchange'] = updated_last_exchange

    # Handle the response based on whether Excel download is selected
    if excel_selected:
        # Assuming the Excel data is already prepared in the generate_response function
        # and stored in the session
        download_url = url_for('download_excel')
        return jsonify({'response': ai_response, 'excelSelected': excel_selected, 'downloadUrl': download_url})
    else:
        # Regular text response
        return jsonify({'response': ai_response, 'excelSelected': excel_selected})


@app.route('/download_excel')
def download_excel():
    # Retrieve the markdown table string stored in the session
    markdown_table_str = session.get('markdown_table_str')
    if not markdown_table_str:
        return "No data available for Excel file", 404

    # Convert the markdown table to a DataFrame
    excel_data = markdown_table_to_dataframe(markdown_table_str)

    # Create a BytesIO buffer
    output = BytesIO()

    # Save DataFrame to Excel in memory
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        excel_data.to_excel(writer, index=False)

    output.seek(0)

    return Response(
        output.getvalue(),
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        headers={'Content-Disposition': 'attachment; filename=report.xlsx'}
    )


def generate_response(question, context, last_exchange, excelSelected, selectedFileId):
    client = openai.OpenAI(api_key='')
    # Modify the question for markdown tabular format if excelSelected is True
    if excelSelected:
        question += "and note : answer this strictly in markdown tabular format so that in can be parsed"

    messages = [{"role": "system", "content": context}]
    if isinstance(last_exchange, list) and last_exchange:
        messages.append(last_exchange[-1])  # Add only the last user message
    messages.append({"role": "user", "content": question})

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    response_text = completion.choices[0].message.content.strip()

    if excelSelected:
        print(response_text)
        # Store the markdown response text in the session for later Excel file generation
        session['markdown_table_str'] = response_text
        return "Your report is ready.", []

    else:
        # Regular text response processing
        formatted_response = convert_text_to_html(response_text)
        updated_last_exchange = [{"role": "user", "content": question},
                                 {"role": "assistant", "content": formatted_response}]
        return formatted_response, updated_last_exchange
        # Modify the format_file_contents function to handle the case where no file is received


def format_file_contents(user_files, selected_file_id):
    formatted_text = "I have these file(s):\n"

    if not selected_file_id:
        # If no file is selected, aggregate all file contents
        for i, file in enumerate(user_files, 1):
            if file.file_content:
                file_content_cleaned = clean_text(file.file_content)
                formatted_text += f"{i}. File Name: {file.file_name} and its content are:\n{file_content_cleaned}\n\n"
    else:
        # Use the content of the selected file
        selected_file = next((file for file in user_files if file.file_id == selected_file_id), None)
        if selected_file and selected_file.file_content:
            formatted_text += f"File Name: {selected_file.file_name}\n{clean_text(selected_file.file_content)}"
        else:
            formatted_text = "Selected file does not exist or has no content."

    return formatted_text


def clean_text(text):
    # Function to clean and structure text
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'[^\w\s\.\,\(\)\-\:]', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


# def generate_response(question, context):
#     client = openai.OpenAI(api_key='')
#
#     completion = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": context},
#             {"role": "user", "content": question}
#         ]
#     )
#
#     # Extracting only the text content from the response
#     # Correctly accessing the content of the message
#     response_text = completion.choices[0].message.content
#     answer = response_text.strip()
#     answer = convert_text_to_html(answer)
#     return answer
#


# def generate_response(question, context):
#     # API URL and key
#     api_key = ""
#     api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
#     query = context + "\n" + question
#
#     # Data to be sent in POST request
#     data = {
#         "contents": [{
#             "parts": [{
#                 "text": query
#             }]
#         }]
#     }
#
#     print(query)
#
#     # Headers for the POST request
#     headers = {
#         'Content-Type': 'application/json'
#     }
#
#     # Making the POST request
#     response = requests.post(api_url, headers=headers, json=data)
#
#     # Checking if the request was successful
#     if response.status_code == 200:
#         # Extract the response text from the JSON data
#         response_text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
#         ai_response_html = convert_markdown_to_html(response_text)
#         return ai_response_html
#     else:
#         # Return an error message if the request failed
#         return f"Error: {response.status_code}, {response.text}"

# def generate_response(question, context):
#     from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
#     from transformers import BertTokenizer, BertForQuestionAnswering
#     import torch
#     import wikipediaapi
#     tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
#     model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
#
#     inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors='pt')
#     input_ids = inputs["input_ids"]
#
#     # Get the predictions for the start and end of the answer
#     with torch.no_grad():  # Disable gradient calculations for inference
#         outputs = model(**inputs)
#         answer_start_scores = outputs.start_logits
#         answer_end_scores = outputs.end_logits
#
#     # Ensure the outputs are tensors before applying argmax
#     if not isinstance(answer_start_scores, torch.Tensor) or not isinstance(answer_end_scores, torch.Tensor):
#         raise TypeError("Model did not return tensor outputs.")
#
#     answer_start = torch.argmax(answer_start_scores)
#     answer_end = torch.argmax(answer_end_scores) + 1
#
#     # Convert tokens to the answer string
#     answer_tokens = input_ids[0, answer_start:answer_end + 1]
#     answer = tokenizer.decode(answer_tokens)
#
#     return answer
