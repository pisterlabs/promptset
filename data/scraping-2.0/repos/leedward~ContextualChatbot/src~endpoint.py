from flask import Flask, request, jsonify
from multiprocessing import Pool
import openai
from chat import create_context, df

from webcrawler import crawl

app = Flask(__name__)


@app.route('/crawl', methods=['POST'])
def crawl_endpoint():
    data = request.get_json()

    if 'url' not in data:
        return {"error": "Missing 'url' in JSON data."}, 400

    url = data['url']
    # Add url to pool for processing
    pool.apply_async(crawl, (url,))
    # crawl(url)

    return {"message": f"Crawling started for {url}"}


@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint to handle chat"""
    if 'question' not in request.json:
        return jsonify({'error': 'Question not provided'}), 400

    question = request.json.get('question')
    context = create_context(question, df)

    try:
        response = openai.ChatCompletion.create(
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": ""},
                {"role": "user", "content": f"Context: {context}"},
                {"role": "user", "content": f"Question: {question}"}
            ],
            model='gpt-3.5-turbo',
            temperature=0.1,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        answer = response['choices'][0]['message']['content'].strip()
        return jsonify({'answer': answer})
    except openai.error.APIError as e:
        print(e)
        return jsonify({'error': 'Error processing your request'}), 500
    except Exception as e:
        print(e)
        return jsonify({'error': 'An unexpected error occurred'}), 500


if __name__ == '__main__':
    pool = Pool(processes=10)
    app.run(port=5000)

