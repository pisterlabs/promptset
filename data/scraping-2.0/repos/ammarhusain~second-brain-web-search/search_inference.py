from flask import Flask, render_template, request
import openai, pinecone
import time, re
import logging, os

app = Flask(__name__)

URL_PREFIX = "https://notes.ammarh.io/"
# initialize connection to pinecone (get API key at app.pinecone.io)
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment="us-east1-gcp"
)
# connect to index
PINECONE_INDEX = pinecone.Index('obsidian-second-brain')

openai.api_key = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = "text-embedding-ada-002"
CONTEXT_LENGTH = 10000
# Landing page
@app.route('/')
def index():
    return render_template('landing.html')

# Second page that displays user input
@app.route('/result', methods=['POST'])
def result():
    start_time = time.time()
    session_variables = {}
    session_variables['searchtext'] = request.form['searchtext']
    session_variables['confidence'] = float(request.form['confidence'])
    session_variables['generate_ans'] = request.form['search-button']

    logging.error(f"session : {session_variables}")
    try:
        search_embedding = openai.Embedding.create(
        input=session_variables['searchtext'],
        engine=EMBED_MODEL
        )['data'][0]['embedding']
    except:
        msg = "OpenAI embedding call failed"
        logging.error(msg)
        return render_template('result.html', results=[], generated_qa=msg, \
                session_variables=session_variables, elapsed_time=time.time()-start_time)
    
    embed_time = time.time() - start_time
    
    try:
        # retrieve from Pinecone
        res = PINECONE_INDEX.query(search_embedding, top_k=20, include_metadata=True)
    except:
        msg = "Pinecone retrieval call failed"
        logging.error(msg)
        return render_template('result.html', results=[], generated_qa=msg, \
                session_variables=session_variables, elapsed_time=time.time()-start_time)

    query_time = time.time() - embed_time

    results = []
    for match in res['matches']:
        if match['score'] < session_variables['confidence']:
            continue
        path_list = match['metadata']['file'].split('/')
        file = path_list[-1] + ' :: ' + '/'.join(path_list[5:-1])
        link = (URL_PREFIX + '/'.join(path_list[6:])).replace(" ", "+")
        filtered_notes = [x for x in match['metadata']['note'].split("\n") \
                            if x != "" and x[0] != "!"]
        context_str = " ".join(filtered_notes)
        results.append({'file': file, 
                        'notes': filtered_notes[:5], 
                        #'notes': match['metadata']['note'].split("\n"),
                        'score': match['score'],
                        'link': link,
                        'context' : context_str
                        })
        
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    results_time = time.time() - query_time

    generated_qa = ""
    generated_qa_time = -1
    if session_variables['generate_ans']:
        context_str = "\n---\n".join([x['context'] for x in results])[:CONTEXT_LENGTH]
        if re.search(r'\w', context_str):
            # build our prompt with the retrieved contexts included
            prompt_start = (
                #"Answer the query based on the context below.\n"+
                "Context:\n"
            )
            prompt_end = (
                f"\n---\nGiven this and only this context elaborate on the query: "+
                f"{session_variables['searchtext']}\nElaborate: "
            )
            prompt = prompt_start + context_str + prompt_end
            try:
                generated_qa = complete_gpt_3_5(prompt)
            except Exception as e:
                msg = f"OpenAI GPT-3.5 text completion failed. Exception - {e}"
                logging.error(msg)
                return render_template('result.html', results=[], generated_qa=msg, \
                        session_variables=session_variables, elapsed_time=time.time()-start_time)
            
            generated_qa_time = time.time() - results_time

    logging.error(f"embed_time={embed_time}, query_time={query_time}, results_time={results_time}, generated_qa_time={generated_qa_time}")
    return render_template('result.html', results=results, generated_qa = generated_qa, \
                session_variables=session_variables, elapsed_time=time.time()-start_time)

def complete(prompt):
    # query text-davinci-003
    res = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        temperature=0,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return res['choices'][0]['text'].strip()

def complete_gpt_3_5(prompt):
    res = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": "You are a helpful assistant that elaborates on the users query primarily using only the context they provide."},
            {"role": "user", "content": prompt}
        ]
    )
    return res['choices'][0]['message']['content']

def complete_gpt_4(prompt):
    res = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
            {"role": "system", "content": "You are a helpful assistant that elaborates on the users query using only the context they provide. If the context does not provide sufficient details for you to formulate an answer you politely let them know."},
            {"role": "user", "content": prompt}
        ]
    )
    return res['choices'][0]['message']['content']



if __name__ == '__main__':
    app.run(debug=True)