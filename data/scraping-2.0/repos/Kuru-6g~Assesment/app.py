from flask import Flask, request, render_template
from openai import OpenAI
from llamaindex import process_pdf  # Import the Llama Index processing function

app = Flask(__name__)

client = OpenAI(api_key="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")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/query-pdf', methods=['POST'])
def query_pdf():
    if 'pdf_file' not in request.files:
        return 'No file part'

    file = request.files['pdf_file']
    question = request.form['question']

    # Check if the uploaded file is a PDF
    if file.filename.endswith('.pdf'):
        try:
            try:
                pdf_text = process_pdf(file.read()).decode('utf-8')
            except UnicodeDecodeError:
                # If decoding as UTF-8 fails, try a different encoding
                try:
                    pdf_text = process_pdf(file.read()).decode(
                        'latin-1')  # You can try 'latin-1' or other encodings here
                except Exception as e:
                    return f'Error decoding PDF: {str(e)}'

        except Exception as e:
            # Handle exceptions when reading the PDF
            return f'Error processing PDF: {str(e)}'

        # Query using OpenAI
        try:
            prompt = f"Context: {pdf_text}\nQuestion: {question}\nAnswer:"
            response = client.completions.create(
                model="curie",
                prompt=prompt
            )
            return response['choices'][0]['text']
        except Exception as e:
            # Handle exceptions when making the API call
            return f'Error querying OpenAI: {str(e)}'



    else:
        return 'Uploaded file is not a PDF'


if __name__ == '__main__':
    app.run(debug=True)
