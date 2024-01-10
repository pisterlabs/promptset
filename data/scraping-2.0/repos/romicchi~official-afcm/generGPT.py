from flask import Flask, request, jsonify
from flask_cors import CORS  # Import the CORS extension
import pymysql
import requests
from decouple import config
import openai
import nltk
from nltk.corpus import stopwords
from pdfminer.high_level import extract_text
import io

app = Flask(__name__)

# Add CORS support to your app
CORS(app)

# Define your database connection parameters
db_config = {
    'host': 'localhost',
    'user': 'u203878552_gener',
    'password': 'Generlnu2023!',
    'db': 'u203878552_laravel_auth',
    'port': 3306
}

# Establish a connection to the database
conn = pymysql.connect(**db_config)

# Function to load PDF URLs from the database
def load_pdf_urls():
    query = "SELECT url FROM Resources"
    with conn.cursor() as cursor:
        cursor.execute(query)
        pdf_urls = [row[0] for row in cursor.fetchall()]
    return pdf_urls

# Download PDF function
def download_pdf(pdf_url):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Failed to download PDF from {pdf_url}: {str(e)}")
        return None

# Load the OpenAI API key from the .env file
openai_api_key = config('OPENAI_API_KEY')

# Set the API key for the OpenAI client
openai.api_key = openai_api_key

# Download NLTK stop words if not already downloaded
nltk.download('stopwords')

# Function to remove stop words
def remove_stop_words(text):
    # Tokenize the text into words
    words = text.split()

    # Get a list of English stop words
    stop_words = set(stopwords.words("english"))

    # Remove stop words from the list of words
    words = [word for word in words if word.lower() not in stop_words]

    # Join the words back into a paragraph
    processed_text = ' '.join(words)
    return processed_text

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('query')

    pdf_urls = load_pdf_urls()  # Load PDF URLs from the database

    # Loop through the PDF URLs
    for pdf_url in pdf_urls:
        pdf_url = pdf_url  # Extract the URL from the result

        # Assuming you have a function to download the PDF from the URL, e.g., download_pdf(pdf_url)
        pdf_content = download_pdf(pdf_url)  # You need to implement this function

        if pdf_content is not None:
            # Extract text from the PDF using pdfminer.six
            with io.BytesIO(pdf_content) as pdf_stream:
                pdf_text = extract_text(pdf_stream)

            # Split the PDF text into paragraphs
            paragraphs = pdf_text.split('\n')

            answers = []  # To store answers

            for paragraph in paragraphs:
                # Remove stop words from each paragraph
                processed_paragraph = remove_stop_words(paragraph)

                # Combine the user-inputted question and context
                context = f"{question}\n{processed_paragraph}"

                # Generate an answer using the ChatGPT 3.5 Turbo API
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": context}
                    ]
                )

                # Extract and store the answer
                answer = response['choices'][0]['message']['content']
                answers.append(answer)

                # Break out of the loop to get only one answer per question
                break

            return jsonify({'answers': answers})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)