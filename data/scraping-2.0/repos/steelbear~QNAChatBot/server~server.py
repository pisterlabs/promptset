import os
from string import Template

import openai
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile

from model import Question, Response
from vectordb import VectorDB

load_dotenv()
client = openai.Client(api_key=os.environ.get('OPENAI_API_KEY'))
vectorDB = VectorDB('QNAChatBot')
app = FastAPI()


prompt = Template('''Analyze and understand the context provided, then offer an answer to the presented question.

Context:
${context}
                  
Question: ${question}

Answer:
- Based on the information in the Context, construct an answer to Question.
- If the information within the context is insufficient to provide a comprehensive answer to the question, respond with 'I'm sorry. There is not enough information to answer this question.'

Analysis and Answer: 
''')


@app.post("/api/ask")
def respond_question(question: Question):
    documents = vectorDB.query(question.filename, question.question, 3)
    documents = [doc_distance[0] for doc_distance in documents] # remove distance

    context = '\n'.join(map(lambda doc: doc.content.replace('\n', ' '), documents)) # remove newline escape in documents and concatenate them

    messages = [
        {
            'role': 'user',
            'content': prompt.substitute(context=context, question=question),
        },
    ]
    
    try:
        result_chat = client.chat.completions.create(
            messages=messages,
            model='gpt-3.5-turbo',
            temperature=0,
        )
        content = result_chat.choices[0].message.content
        error = None
    except openai.APIStatusError as e:
        content = ''
        error = f"Got error from OpenAI API:\n  {e.status_code} {e.response}"
    except openai.APIConnectionError as e:
        content = ''
        error = f"Got error from OpenAI API:\n  {e.__cause__}"
    return Response(content=content, error=error)


@app.post("/api/upload")
def upload_file(file: UploadFile):
    vectorDB.upload_pdf(file.file, file.filename)
    return Response(content=file.filename)