import os
from dotenv import load_dotenv
import requests
import sqlite3
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
import openai
from werkzeug.utils import secure_filename
import PyPDF2
from docx import Document
from bs4 import BeautifulSoup, NavigableString, Tag

# from web_crawler import scrape_page

# Configurations
app = Flask(__name__)
flask_secret_key = os.environ.get("FLASK_SECRET_KEY")
app.config["UPLOAD_FOLDER"] = "uploads"
openai.api_key = os.environ.get("OPENAI_API_KEY")
Authorization: os.environ.get("OPENAI_API_KEY")
ALLOWED_EXTENSIONS = {'pdf', 'docx'}

# Database
def create_connection(db_name):
    conn = sqlite3.connect(db_name)
    return conn

conn1 = create_connection("chatbot_data.db")
conn2 = create_connection("conversations.db")

def create_conversations_table(conn):
    # Your table creation code goes here

# Call the create_conversations_table function
 def create_conversations_table(conn):
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY,
        user_message TEXT NOT NULL,
        chatbot_response TEXT NOT NULL
    )
    """)
    conn.commit()

# Call the create_conversations_table function
create_conversations_table(conn2)

conn2.close()

# Routes
@app.route("/")
def index():
    return render_template("chatbot.html")

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['message']

    # Check if the user message contains the 'scrape' command
    if user_message.lower().startswith('scrape'):
        url = user_message.split(' ', 1)[1].strip()  # Extract the URL from the message
        extracted_text = scrape_page(url)  # Scrape the text from the webpage
        prompt = f"{extracted_text} {user_message}"  # Add the extracted text to the GPT-3.5 prompt
    else:
        prompt = user_message

    try:
        chatbot_response = get_chatbot_response(prompt)  # Call GPT-3.5 API
    except Exception as e:
        print(f"Error generating response: {e}")
        chatbot_response = "Sorry, I couldn't generate a response. Please try again later."

    try:
        # Store conversation data
        insert_conversation(conn2, user_message, chatbot_response)
    except Exception as e:
        print(f"Error inserting conversation data: {e}")

    return {'response': chatbot_response}

def get_chatbot_response(message):
    # Set up the OpenAI API call
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": message}],
        max_tokens=150,
        temperature=0.5,
    )

    # Extract the generated response text
    response_text = response.choices[0].message.content.strip()
    return response_text

@app.route("/upload", methods=["POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["file"]
        category = request.form["category"]

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            
            try:
                # Process the file, extract content, and store it in the database
                content = extract_content(os.path.join(app.config["UPLOAD_FOLDER"], filename))
                insert_document(conn1, category, content)

                flash("File uploaded and processed successfully.")
            except Exception as e:
                print(f"Error processing and storing the file: {e}")
                flash("There was an error processing your file. Please try again later.")
            
            return redirect(url_for("index"))
        else:
            flash("File format not allowed. Please upload a PDF or DOCX file.")
            return redirect(url_for("index"))
@app.route('/upload')
def upload():
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)

# GPT-3.5 API
def generate_response(prompt):
    documents = get_all_documents(conn1,conn2)
    context = ' '.join(documents)  # Concatenate all document contents
    prompt_with_context = f"{context} {prompt}"
    
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=prompt_with_context,
        max_tokens=50,  # Adjust this value according to your needs
        n=1,
        stop=None,
        temperature=0.5,
    )

    return response.choices[0].text.strip()

# SQLite functions
def create_table(conn):
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message TEXT NOT NULL,
            chatbot_response TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

def create_conversations_table(conn):
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY,
        user_message TEXT NOT NULL,
        chatbot_response TEXT NOT NULL
    )
    """)
    conn.commit()

def insert_conversation(conn, user_message, chatbot_response):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO conversations (user_message, chatbot_response)
        VALUES (?, ?)
    """, (user_message, chatbot_response))
    conn.commit()

def create_documents_table(conn):
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL
        )
    """)
    conn.commit()

create_documents_table(conn1)

def insert_document(conn, title, content):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO documents (title, content)
        VALUES (?, ?)
    """, (title, content))
    conn.commit()

def get_all_documents(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT content FROM documents")
    documents = cursor.fetchall()
    return [doc[0] for doc in documents]

# File handling
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_content(filepath):
    file_ext = filepath.rsplit(".", 1)[1].lower()
    content = ""

    if file_ext == "pdf":
        with open(filepath, "rb") as file:
            pdf_reader = PyPDF2.PdfFileReader(file)
            for page_num in range(pdf_reader.getNumPages()):
                content += pdf_reader.getPage(page_num).extractText()
    elif file_ext == "docx":
        doc = Document(filepath)
        for para in doc.paragraphs:
            content += para.text

    return content

# Truncate text
def truncate_text(text, max_characters, preserve_words=True, ellipsis="..."):
    text = text.strip()

    if len(text) <= max_characters:
        return text

    if preserve_words:
        truncated_text = text[:max_characters].rsplit(" ", 1)[0]
    else:
        truncated_text = text[:max_characters]

    if len(truncated_text) + len(ellipsis) <= max_characters:
        truncated_text += ellipsis
    else:
        truncated_text = truncated_text[:-len(ellipsis)] + ellipsis

    return truncated_text

# Web scraper
def clean_text(text):
    return ' '.join(text.split())

def extract_text(element):
    if isinstance(element, NavigableString):
        return clean_text(element.string)
    if isinstance(element, Tag):
        text = []
        for child in element.children:
            child_text = extract_text(child)
            if child_text:
                text.append(child_text)
        return ' '.join(text)
    return ''

def scrape_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Remove unwanted elements (e.g., scripts, styles)
    for unwanted in soup(['script', 'style']):
        unwanted.decompose()

    # Extract text from relevant tags
    relevant_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'strong', 'em', 'a', 'span']
    extracted_text = ' '.join([extract_text(tag) for tag in soup.find_all(relevant_tags)])

    return extracted_text
