from flask import Flask, request, jsonify
import requests
import sentry_sdk
import glob
from bs4 import BeautifulSoup
import os
from sentry_sdk.integrations.flask import FlaskIntegration
from flask_jwt_extended import create_access_token,get_jwt,get_jwt_identity, \
                               unset_jwt_cookies, jwt_required, JWTManager
from sentry_sdk import set_user
from datetime import datetime,timedelta, timezone
from langchain.document_loaders import UnstructuredFileLoader
from werkzeug.utils import secure_filename
import re

app = Flask(__name__)
app.config["JWT_SECRET_KEY"] = "solata-je-najboljša"
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=20)
app.config['PROPAGATE_EXCEPTIONS'] = True

jwt = JWTManager(app)

sentry_sdk.init(
    dsn="INSERT OWN API KEY",
    integrations=[
        FlaskIntegration(),
    ],

    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    # We recommend adjusting this value in production.
    traces_sample_rate=1.0,
    send_default_pii=True
)

def scrape(urls):
    aggregated_text = ""
    for url in urls:
        response = requests.get(url)
        response.raise_for_status()


        # Parse the webpage content with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove all script and style elements
        for script in soup(["script", "style"]):
            script.extract()  # Removes the tag from the tree

        # Extract the text from the parsed content
        text = soup.get_text()

        cleaned_text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        aggregated_text += cleaned_text + "\n"

    return aggregated_text

def process_file(filepath, user_prefix, name):
    loader = UnstructuredFileLoader(filepath)
    docs = loader.load()

    output_filename = f"{name}.txt"
    user_specific_dir = os.path.join("podatki-files", user_prefix)

    # Ensure the directory exists
    if not os.path.exists(user_specific_dir):
        os.makedirs(user_specific_dir)

    output_filepath = os.path.join(user_specific_dir, output_filename)

    # Saving the data to the new .txt file
    with open(output_filepath, 'w', encoding='utf-8') as out_file:
        for doc in docs:
            # Use regex to replace multiple newlines with a single one
            cleaned_content = re.sub(r'\n+', '\n', doc.page_content)
            out_file.write(cleaned_content)

    return output_filepath

def process_file_word_pdf(filepath):
    loader = UnstructuredFileLoader(filepath)
    docs = loader.load()
    aggregated_text = ""
    for doc in docs:
        # Use regex to replace multiple newlines with a single one
        cleaned_content = re.sub(r'\n+', '\n', doc.page_content)
        aggregated_text += cleaned_content + "\n"
    return aggregated_text  # Return the processed text instead of writing to a file


@app.route('/scrape', methods=['GET', 'POST'])
@jwt_required()
def handle_scrape_request():
    with sentry_sdk.start_transaction(name="scrape", op="endpoint"):
        try:
            urls = request.args.getlist('url')
            name = request.args.get('name')

            #if not urls:
            #    return jsonify({'error': 'URL parameter is required'}), 400
            
            client_ip = request.remote_addr
            set_user({
                "email": get_jwt_identity(),
                "ip_address": client_ip
            })


            aggregated_text = ""
            
            # Handle file uploads
            uploaded_files = request.files.getlist('file')
            for uploaded_file in uploaded_files:
                if uploaded_file.filename != '':
                    file_path = secure_filename(uploaded_file.filename)
                    uploaded_file.save(file_path)
                    processed_text = process_file_word_pdf(file_path)
                    aggregated_text += processed_text + "\n"
                    os.remove(file_path)


        

            if urls:
                scraped_text = scrape(urls)
                aggregated_text += scraped_text + "\n"
            user_email = get_jwt_identity()
            user_prefix = user_email.split('@')[0]

            user_specific_dir = os.path.join("podatki-link", user_prefix)

            # Delete all txt files in the directory if it exists
            if os.path.exists(user_specific_dir):
                txt_files = glob.glob(os.path.join(user_specific_dir, "*.txt"))  # Get a list of all txt files in the directory
                for txt_file in txt_files:
                    os.remove(txt_file)  # Delete each txt file

            # Ensure the directory exists
            else:
                os.makedirs(user_specific_dir)

            filename = f"{name}.txt"
            filepath = os.path.join(user_specific_dir, filename)

            # Write the scraped text to the file (no need to check if the file exists as we have just deleted all txt files)
            with open(filepath, 'w', encoding='utf-8') as file:
                file.write(aggregated_text)
        


            return jsonify({'message': 'Scraping and saving completed successfully', 'filename': filename})


        except requests.HTTPError as http_err:
            return jsonify({'error': f'HTTP error occurred: {http_err}'}), 500
        except Exception as err:
            return jsonify({'error': f'An error occurred: {err}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port='5052')
