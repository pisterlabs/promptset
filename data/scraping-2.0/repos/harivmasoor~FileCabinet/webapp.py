from flask import Flask, request, render_template, redirect, url_for, flash 
from uuid import uuid4
import openai
import pinecone
from pypdf import PdfReader
from io import BytesIO
from dotenv import load_dotenv
import os
from flask import session
load_dotenv()



app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')


# OpenAI API Key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Pinecone Setup
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))
pinecone_index = pinecone.Index("pdf-embeddings")
pdf_data_store = {}
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        pdfs = request.files.getlist('pdf')
        # Generate unique IDs
        doc_ids = [str(uuid4()) for _ in range(len(pdfs))]

        # Extract text and get embeddings
        vectors_to_upsert = []
        for doc_id, pdf in zip(doc_ids, pdfs):
            reader = PdfReader(BytesIO(pdf.read()))
            text = " ".join([page.extract_text() for page in reader.pages])
            pdf_data_store[doc_id] = text 
            embedding = get_openai_embedding(text)

            # Ensure embedding is a list of floats
            if isinstance(embedding, str):
                embedding = [float(val) for val in embedding.split(',')]  # Adjust as per the actual format

            vectors_to_upsert.append({
                'id': doc_id,
                'values': embedding,
                'metadata': {
                    'full_text': text,  # Store the first 1000 characters as an example
                    # Or you could store the entire text if desired:
                    # 'full_text': text
                }
            })

        # Upsert to Pinecone
        pinecone_index.upsert(vectors=vectors_to_upsert)

        flash('PDFs uploaded and indexed!', 'success')
        return redirect(url_for('index'))

    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():

  query = request.form['query']
  embedding = get_openai_embedding(query)  

  results = pinecone_index.query(queries=[embedding], top_k=50)
  return render_template('results.html', results=results)

def get_openai_embedding(text):
  response = openai.Embedding.create(
    input=[text],
    model="text-embedding-ada-002", 
    return_embeddings=True
  )
  return response['data'][0]['embedding'] 

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['message']
    embedding = get_openai_embedding(user_message)
    
    results = pinecone_index.query(queries=[embedding], top_k=5, include_metadata=True)
    print(results)

    if results['results'] and results['results'][0]['matches']:
        top_match = results['results'][0]['matches'][0]
        matched_metadata = top_match.get('metadata', {})
        matched_text = matched_metadata.get('full_text', 'No matched text found')
        # Now, query GPT to get a response
        gpt_response = ask_gpt(user_message, matched_text)
    else:
        matched_text = "No matches found"
        gpt_response = f"No matches found for '{user_message}'. Please refine your query."

    # Store GPT response and user message in the session for chat history
    if 'chat_history' not in session:
        session['chat_history'] = []
    session['chat_history'].extend([{"user": user_message}, {"gpt": gpt_response}])

    return render_template('index.html', chat_history=session['chat_history'])

def ask_gpt(user_message, context):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"The user is asking: {user_message}"},
        {"role": "assistant", "content": f"Here's a matched document excerpt: {context}"}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    # Extract the assistant's message from the response
    gpt_response = response.choices[0].message['content']
    return gpt_response



if __name__ == '__main__':
  app.run(debug=True)