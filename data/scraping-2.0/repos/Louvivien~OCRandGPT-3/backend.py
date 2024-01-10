"""My Python module.

This module contains functions and classes for performing various tasks.
"""

# Module code goes here
import sys
import fitz
from flask import Flask, render_template, request
import tempfile
import openai

sys.path.append('templates/frontend/module')


app = Flask(__name__, template_folder='templates')

# Set your OpenAI API key
openai.api_key = "xxxxxx"

@app.route('/', methods=['GET', 'POST'])
def extract_pdf_text():
    """
    Extracts text from specific pages of a PDF file and returns it as a string.

    Args:
        pdf_file (FileStorage): The uploaded PDF file.
        page_numbers (list): A list of integers representing the page numbers to extract.

    Returns:
        str: The extracted text.
    """
    if request.method == 'POST':
        # Get the uploaded PDF file and page numbers from the form data
        pdf_file = request.files['file']
        page_numbers = request.form['page_numbers']
        # Split the page numbers by comma and convert them to integers
        page_numbers = [int(x) for x in page_numbers.split(',')]
        
        # Save the PDF file to a temporary location on the filesystem
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = f"{temp_dir}/temp.pdf"
            pdf_file.save(pdf_path)
    
            # Open the PDF file using PyMuPDF
            pdf_doc = fitz.Document(pdf_path)

        # Initialize an empty list to store the extracted text
        extracted_text = []

        # Iterate through the specified page numbers
        for page_number in page_numbers:
            # Make sure the page number is within the range of the PDF
            if page_number > 0 and page_number <= pdf_doc.page_count:
                # Load the page from the PDF document
                page = pdf_doc.load_page(page_number - 1)
                # Extract the text from the page
                text = page.get_text()
                # Add the extracted text to the list
                extracted_text.append(text)
            else:
                # Add an error message to the list if the page number is out of range
                extracted_text.append(f"Page {page_number} is not a valid page number.")
            
            # Send the extracted text to ChatGPT and get the response
            model = "text-davinci-003"
            prompt = "summarize".join(extracted_text)
            
            response = openai.Completion.create(
                engine=model,
                prompt=prompt,
                max_tokens=150,
                n=1,
                stop=None,
                temperature=0.5,
                presence_penalty=1.0,
                top_p=1.0,
            )

            chatbot_response = response.choices[0].text
                
            # Render the HTML template with the extracted text and ChatGPT response
            return render_template('frontend.html', extracted_text=extracted_text, chatbot_response=chatbot_response)
    else:
        # Render the HTML template for the GET request
        return render_template('frontend.html')


if __name__ == '__main__':
    app.run(debug=True)
