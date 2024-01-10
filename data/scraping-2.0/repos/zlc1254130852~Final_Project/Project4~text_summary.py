from flask import Blueprint
from flask import render_template
from flask import request, Response
from AI_chat import openai_client
from login_check import check_login
import json,time
import PyPDF2
from docx import Document


text_summary_bp = Blueprint('text_summary', __name__)

@text_summary_bp.route('/text_summary', methods=['GET'])
def text_summary():
    user_info = check_login()  # check which user is logged in.

    if user_info:  # if there is a logged-in user
        return render_template("text_summary.html", current_user=user_info.login_name)
    else:
        return render_template("text_summary.html")

@text_summary_bp.route('/file3',methods=['POST'])
def save_file():
    data = request.files

    file = data['file']

    form = request.form
    current_user = form.get('user')

    buffer_data = file.read()

    if len(file.filename)>=4 and file.filename[-4:]==".pdf":
        with open("static/en/" + "pdf" + current_user + file.filename[-4:], 'wb+') as f:
            f.write(buffer_data)

        f.close()

        pdf_file = open("static/en/" + "pdf" + current_user + file.filename[-4:], 'rb')

        pdf_reader = PyPDF2.PdfReader(pdf_file)

        pages_num = len(pdf_reader.pages)

        with open("static/en/" + 'pdf2txt'+current_user+'.txt', 'w') as txt_file:
            for page_index in range(pages_num):
                page_content = pdf_reader.pages[page_index].extract_text()
                txt_file.write(page_content)

        pdf_file.close()
        txt_file.close()

        with open("static/en/" + 'pdf2txt' + current_user + '.txt', 'r') as txt_file:
            buffer_data = txt_file.read()
        txt_file.close()

    elif len(file.filename)>=5 and file.filename[-5:]==".docx":

        with open("static/en/" + "docx" + current_user + file.filename[-5:], 'wb+') as f:
            f.write(buffer_data)

        f.close()

        doc = Document("static/en/" + "docx" + current_user + file.filename[-5:])

        with open("static/en/" + 'docx2txt' + current_user + '.txt', 'w', encoding='utf-8') as f:
            for paragraph in doc.paragraphs:
                f.write(paragraph.text + '\n')

        f.close()

        with open("static/en/" + 'docx2txt' + current_user + '.txt', 'r') as txt_file:
            buffer_data = txt_file.read()
        txt_file.close()

    elif len(file.filename)>=4 and file.filename[-4:]==".txt":

        buffer_data = buffer_data.decode('utf-8')

    print(buffer_data)

    response = openai_client[current_user].chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a highly skilled AI trained in language comprehension and summarization. I would like you to read the following text and summarize it into a concise abstract paragraph. Aim to retain the most important points, providing a coherent and readable summary that could help a person understand the main points of the discussion without needing to read the entire text. Please avoid unnecessary details or tangential points."
            },
            {
                "role": "user",
                "content": buffer_data
            }
        ],
        stream=True
    )
    response2 = openai_client[current_user].chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a proficient AI with a specialty in distilling information into key points. Based on the following text, identify and list the main points that were discussed or brought up. These should be the most important ideas, findings, or topics that are crucial to the essence of the discussion. Your goal is to provide a list that someone could read to quickly understand what was talked about."
            },
            {
                "role": "user",
                "content": buffer_data
            }
        ],
        stream=True
    )

    def generate():
        for trunk in response:
            print(json.dumps(
                {'delta': trunk.choices[0].delta.content, 'finish_reason': trunk.choices[0].finish_reason}))
            time.sleep(0.1)
            yield json.dumps(
                {'delta': trunk.choices[0].delta.content, 'finish_reason': trunk.choices[0].finish_reason})+"\n"
        for trunk in response2:
            print(json.dumps(
                {'delta': trunk.choices[0].delta.content, 'finish_reason': trunk.choices[0].finish_reason}))
            time.sleep(0.1)
            yield json.dumps(
                {'delta': trunk.choices[0].delta.content, 'finish_reason': trunk.choices[0].finish_reason})+"\n"

    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no'
    }

    return Response(generate(), mimetype="text/event-stream", headers=headers)

@text_summary_bp.route('/summarize', methods=['POST'])
def summarize():
    response = openai_client[request.json["current_user"]].chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a highly skilled AI trained in language comprehension and summarization. I would like you to read the following text and summarize it into a concise abstract paragraph. Aim to retain the most important points, providing a coherent and readable summary that could help a person understand the main points of the discussion without needing to read the entire text. Please avoid unnecessary details or tangential points."
            },
            {
                "role": "user",
                "content": request.json["text"]
            }
        ],
        stream=True
    )
    response2 = openai_client[request.json["current_user"]].chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a proficient AI with a specialty in distilling information into key points. Based on the following text, identify and list the main points that were discussed or brought up. These should be the most important ideas, findings, or topics that are crucial to the essence of the discussion. Your goal is to provide a list that someone could read to quickly understand what was talked about."
            },
            {
                "role": "user",
                "content": request.json["text"]
            }
        ],
        stream=True
    )
    def generate():
        for trunk in response:
            print(json.dumps(
                {'delta': trunk.choices[0].delta.content, 'finish_reason': trunk.choices[0].finish_reason}))
            time.sleep(0.1)
            yield json.dumps(
                {'delta': trunk.choices[0].delta.content, 'finish_reason': trunk.choices[0].finish_reason})+"\n"
        for trunk in response2:
            print(json.dumps(
                {'delta': trunk.choices[0].delta.content, 'finish_reason': trunk.choices[0].finish_reason}))
            time.sleep(0.1)
            yield json.dumps(
                {'delta': trunk.choices[0].delta.content, 'finish_reason': trunk.choices[0].finish_reason})+"\n"

    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no'
    }

    return Response(generate(), mimetype="text/event-stream", headers=headers)