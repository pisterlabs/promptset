from flask import Blueprint, request, jsonify, Response
import os
import json
import base64
import uuid
import re
from dotenv import load_dotenv
from openai import OpenAI
from google.cloud import storage
from google.cloud import tasks_v2
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import *
from utils import parse_excel, generate_excel

load_dotenv()

storage_client = storage.Client()

openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_api_key)

db = firestore.Client(project='withcontextai')
api_url = 'https://openai-tools-mmxbwgwwaq-uw.a.run.app'
# api_url = 'https://local.lililulu.com'
from_email_text = os.environ.get("FROM_EMAIL")


def format_data(data):
    formatted_data = [
        [[key, str(row[key])] for key in row.keys()] for row in data
    ]
    return formatted_data


def create_sheets(formatted_data):
    sheets = [{'id': f'{uuid.uuid4()}', 'row': row} for row in formatted_data]
    return sheets


def create_questions(sheets, user_message):
    questions = []
    for sheet in sheets:
        text = user_message.strip()
        for key, value in sheet['row']:
            text = re.sub(r'{%s}' % re.escape(key), value, text)
        questions.append({
            'id': sheet['id'],
            'text': text,
        })
    return questions


main_routes = Blueprint('main_routes', __name__)


@main_routes.route('/result', methods=['GET'])
def result_route():
    request_id = request.args.get('request_id')
    if request_id:
        requests_data = db.collection('requests').document(
            request_id).get().to_dict()
        qna_ref = db.collection('qna').where(
            filter=FieldFilter("request_id", "==", request_id))
        qna_data = []
        for doc in qna_ref.stream():
            qna_data.append({"id": doc.id, **doc.to_dict()})
        return jsonify({"requests": requests_data, "qna": qna_data})
    else:
        requests_ref = db.collection('requests').order_by(
            'created_at', direction=firestore.Query.DESCENDING).limit(10)
        requests_data = []
        for doc in requests_ref.stream():
            requests_data.append({"id": doc.id, "doc": doc.to_dict()})
        return jsonify(requests_data)


@main_routes.route('/parse_excel', methods=['POST'])
def parse_excel_route():
    # 检查是否上传了文件
    if 'file' not in request.files:
        return 'No file uploaded'

    file = request.files['file']

    # 检查文件是否符合要求
    if file.filename == '':
        return 'No file selected'
    if not file.filename.endswith('.xlsx'):
        return 'Invalid file type'

    # 调用解析函数
    json_data = parse_excel(file)

    # 返回 JSON 格式数据
    return jsonify(json_data)


@main_routes.route('/upload_excel', methods=['POST'])
def upload_excel_route():
    # 检查是否上传了文件
    if 'file' not in request.files:
        return 'No file uploaded'

    file = request.files['file']

    # 检查文件是否符合要求
    if file.filename == '':
        return 'No file selected'
    if not file.filename.endswith('.xlsx'):
        return 'Invalid file type'

    # 调用解析函数
    json_data = parse_excel(file)

    formatted_data = format_data(json_data)
    sheets = create_sheets(formatted_data)
    upload_data = json.dumps(sheets)

    # Upload the JSON data to the bucket
    bucket_name = 'openai-tools'
    bucket = storage_client.get_bucket(bucket_name)
    random_uuid = uuid.uuid4()
    blob_name = f'{random_uuid}.txt'
    blob = bucket.blob(blob_name)
    blob.upload_from_string(upload_data)

    # 返回 JSON 格式数据
    data = {
        'blob_name': blob_name,
        'json_data': json_data,
    }
    return jsonify(data)


@main_routes.route('/ask_all_questions', methods=['POST'])
def ask_all_questions_route():
    # Get the Authorization header value
    auth_header = request.headers.get("Authorization")
    # Check if the header value exists
    if not auth_header:
        return jsonify({"error": "Authorization header is required"}), 401
    # Extract the token by splitting the header value by whitespace (assuming "Bearer" scheme)
    auth_token = auth_header.split(" ")[1]
    is_auth_token_valid = auth_token == os.environ.get("ACCESS_CODE")
    if not is_auth_token_valid:
        return jsonify({"error": "Authorization is not valid"}), 403

    data = request.get_json()
    config = data.get("config", {})
    email = data.get('email', [])
    excel_blob_name = data.get('blob_name', '')

    bucket_name = 'openai-tools'
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(excel_blob_name)
    sheets_string = blob.download_as_string().decode('utf-8')
    sheets = json.loads(sheets_string)
    user_message = config.get('userMessage', [])
    questions = create_questions(sheets, user_message)

    print("email", email)
    print("config", config)

    request_data = {
        "email": email,
        "config": config,
        "excel_blob_name": excel_blob_name,
        "questions_count": len(questions),
        "success_count": 0,
        "fail_count": 0,
        "created_at": firestore.SERVER_TIMESTAMP,
    }
    _, request_ref = db.collection('requests').add(request_data)

    request_id = request_ref.id
    print("request_id", request_id)

    tasks_client = tasks_v2.CloudTasksClient()
    parent = tasks_client.queue_path(
        'withcontextai', 'us-west1', 'chat-completions-queue')

    for _, question in enumerate(questions):
        question_id = question.get("id")
        question_text = question.get("text")

        payload = json.dumps({
            "request_id": request_id,
            "question_id": question_id,
            "question_text": question_text
        })

        task = {
            'http_request': {
                'http_method': 'POST',
                'url': api_url + '/chat_completions_async',
                'headers': {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer ' + auth_token,
                },
                'body': payload.encode(),
            }
        }

        tasks_client.create_task(request={'parent': parent, 'task': task})

    sg = SendGridAPIClient(api_key=os.environ.get('SENDGRID_API_KEY'))

    # 创建邮件
    from_email = From(from_email_text)  # 发件人
    to_email = To(email)  # 收件人
    subject = "批量任务: " + request_id
    body = Content(
        "text/plain", "任务正在运行中，稍后邮件通知您结果。若长时间未收到邮件，可使用 request_id: " + request_id + " 手动发送。")
    mail = Mail(from_email, to_email, subject, body)

    try:
        # 发送邮件
        response = sg.send(mail)
        if response.status_code != 202:  # 如果发送邮件失败，返回报错
            return {"error": f"Failed to send email, error code: {response.status_code}"}, 400
    except Exception as e:
        return {"error": f"Failed to send email: {e}"}, 400

    print("The confirm email has sent to: ", email)
    return jsonify({"success": True, "message": "Tasks created successfully", "request_id": request_id}), 200


@main_routes.route('/chat_completions_async', methods=['POST'])
def chat_completions_async_route():
    # Get the Authorization header value
    auth_header = request.headers.get("Authorization")

    # Check if the header value exists
    if not auth_header:
        return jsonify({"error": "Authorization header is required"}), 401

    # Extract the token by splitting the header value by whitespace (assuming "Bearer" scheme)
    auth_token = auth_header.split(" ")[1]

    is_auth_token_valid = auth_token == os.environ.get("ACCESS_CODE")
    if not is_auth_token_valid:
        return jsonify({"error": "Authorization is not valid"}), 403

    if not request.is_json:
        return jsonify({"error": "JSON data expected"}), 400

    data = request.get_json()
    request_id = data.get("request_id")
    question_id = data.get("question_id")
    question_text = data.get("question_text")

    if not request_id or not question_id or not question_text:
        return jsonify({"error": "Data missing: request_id, question_id, or question_text"}), 400

    request_data = db.collection('requests').document(
        request_id).get().to_dict()
    config = request_data.get("config", {})
    model = config.get("model", "gpt-3.5-turbo")
    system_message = config.get(
        "system", "You are ChatGPT, a large language model trained by OpenAI.")
    messages = [{"role": "system", "content": system_message},
                {"role": "user", "content": question_text}]
    temperature = config.get("temperature", 0.7)
    presence_penalty = config.get("presence_penalty", 0)
    frequency_penalty = config.get("frequency_penalty", 0)

    if not (model and messages):
        return jsonify({"error": "model and messages must be provided"}), 400

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
        db.collection('requests').document(request_id).update(
            {"success_count": firestore.Increment(1)})
    except Exception as e:
        db.collection('requests').document(request_id).update(
            {"fail_count": firestore.Increment(1)})
        return jsonify({"error": str(e)}), 500

    answer_text = response.choices[0].message.content

    db.collection('qna').add({
        "request_id": request_id,
        "question_id": question_id,
        "question_text": question_text,
        "answer_text": answer_text,
    })

    print("question_id: ", question_id)
    print("answer_text: ", answer_text)

    request_data = db.collection('requests').document(
        request_id).get().to_dict()
    questions_count = request_data.get("questions_count", 0)
    success_count = request_data.get("success_count", 0)
    print(questions_count, success_count)

    if request_data is not None and success_count == questions_count:
        payload = json.dumps({
            "request_id": request_id,
        })

        task = {
            'http_request': {
                'http_method': 'POST',
                'url': api_url + '/send_answers_email',
                'headers': {
                    'Content-Type': 'application/json'
                },
                'body': payload.encode(),
            }
        }

        tasks_client = tasks_v2.CloudTasksClient()
        parent = tasks_client.queue_path(
            'withcontextai', 'us-west1', 'send-answers-email-queue')
        tasks_client.create_task(request={'parent': parent, 'task': task})

    return jsonify({"message": f"Answer published for question {question_id}"}), 200


@main_routes.route('/send_answers_email', methods=['POST'])
def send_answers_email_route():
    data = request.get_json()

    # 从请求中获取 request_id
    request_id = data.get("request_id")

    request_data = db.collection('requests').document(
        request_id).get().to_dict()
    email = request_data["email"]
    excel_blob_name = request_data["excel_blob_name"]

    bucket_name = 'openai-tools'
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(excel_blob_name)
    sheets_string = blob.download_as_string().decode('utf-8')
    sheets = json.loads(sheets_string)

    qna_ref = db.collection('qna').where(
        filter=FieldFilter("request_id", "==", request_id))
    answers = []
    for doc in qna_ref.stream():
        answer = doc.to_dict()
        answers.append({"id": answer.get("question_id"),
                       "text": answer.get("answer_text")})

    # 整合 sheets 和 answers 数据
    for sheet in sheets:
        for answer in answers:
            if sheet["id"] == answer["id"]:
                sheet["row"].append(["answer", answer["text"]])
    json_data = [item["row"] for item in sheets]

    # 生成 excel 表格并保存到内存
    output = generate_excel(json_data)

    # 使用 base64 对 xlsx 文件进行编码
    data = output.read()
    encoded_data = base64.b64encode(data).decode()

    sg = SendGridAPIClient(api_key=os.environ.get('SENDGRID_API_KEY'))

    # 创建邮件附件
    attachment = Attachment()
    attachment.file_content = FileContent(encoded_data)
    attachment.file_type = FileType(
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    attachment.file_name = FileName('output.xlsx')
    attachment.disposition = Disposition('attachment')

    # 创建邮件
    from_email = From(from_email_text)  # 发件人
    to_email = To(email)  # 收件人
    subject = "批量任务: " + request_id
    body = Content("text/plain", "运行结果见附件。")
    mail = Mail(from_email, to_email, subject, body)
    mail.add_attachment(attachment)

    try:
        # 发送邮件
        response = sg.send(mail)
        if response.status_code != 202:  # 如果发送邮件失败，返回报错
            return {"error": f"Failed to send email, error code: {response.status_code}"}, 400
    except Exception as e:
        return {"error": f"Failed to send email: {e}"}, 400

    print("The result email has sent to: ", email)
    return {"message": "Email sent successfully"}, 200


@main_routes.route('/chat_completions', methods=['POST'])
def chat_completions_route():
    # Get the Authorization header value
    auth_header = request.headers.get("Authorization")

    # Check if the header value exists
    if not auth_header:
        return jsonify({"error": "Authorization header is required"}), 401

    # Extract the token by splitting the header value by whitespace (assuming "Bearer" scheme)
    auth_token = auth_header.split(" ")[1]

    is_auth_token_valid = auth_token == os.environ.get("ACCESS_CODE")
    if not is_auth_token_valid:
        return jsonify({"error": "Authorization is not valid"}), 403

    if not request.is_json:
        return jsonify({"error": "JSON data expected"}), 400

    data = request.get_json()

    model = data.get("model", "gpt-3.5-turbo")
    messages = data.get("messages", [])
    temperature = data.get("temperature", 0.7)
    presence_penalty = data.get("presence_penalty", 0)
    frequency_penalty = data.get("frequency_penalty", 0)

    if not (model and messages):
        return jsonify({"error": "model and messages must be provided"}), 400

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@main_routes.route('/generate_excel', methods=['POST'])
def generate_excel_route():
    data = request.get_json()

    # 从请求中获取 request_id
    request_id = data.get("request_id")

    request_data = db.collection('requests').document(
        request_id).get().to_dict()
    email = request_data["email"]
    excel_blob_name = request_data["excel_blob_name"]

    bucket_name = 'openai-tools'
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(excel_blob_name)
    sheets_string = blob.download_as_string().decode('utf-8')
    sheets = json.loads(sheets_string)

    qna_ref = db.collection('qna').where(
        filter=FieldFilter("request_id", "==", request_id))
    answers = []
    for doc in qna_ref.stream():
        answer = doc.to_dict()
        answers.append({"id": answer.get("question_id"),
                       "text": answer.get("answer_text")})

    # 整合 sheets 和 answers 数据
    for sheet in sheets:
        for answer in answers:
            if sheet["id"] == answer["id"]:
                sheet["row"].append(["answer", answer["text"]])
    json_data = [item["row"] for item in sheets]

    # 生成 excel 表格并保存到内存
    output = generate_excel(json_data)
    # 将 xlsx 文件作为响应发送给客户端
    response = Response(output.read(
    ), mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response.headers.set('Content-Disposition',
                         'attachment', filename='output.xlsx')
    return response
