from flask import Flask, request, render_template, session
from langchain import OpenAI
from llama_index.readers import ChatGPTRetrievalPluginReader
from llama_index import ListIndex, LLMPredictor, ServiceContext, PromptHelper
import requests
from datetime import datetime
import pymysql
import os
bearer_token = os.getenv("BEARER_TOKEN")
set_bearer_token = 'Bearer ' + bearer_token

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 세션을 암호화하기 위한 시크릿 키

# 메인 화면
@app.route('/')
def index_page():
    session['chat_list'] = []
    session.modified = True
    return render_template('index.html')

# 관리 화면
@app.route('/manage')
def manage_page():
    return render_template('manage.html')

# 목록 요청
@app.route('/send_list', methods=['POST'])
def send_list():
    category = request.form.get('select')
    result = select_from_table(category)
    model_name = select_from_model(category)
    if model_name and model_name[0]:
        model_name = model_name[0][0]
    else:
        model_name = None
    print(model_name)
    return render_template('manage.html', items=result, category=category, model=model_name)

# 폴더 내 파일들 전부 upsert 요청
@app.route('/send_dir_upsert', methods=['POST'])
def send_dir_upsert():
    # 챗 봇 형식으로 대화내용 저장하기 위한 세션세팅
    session['chat_list'] = []
    session.modified = True
    # 기존에 올라와있는 데이터 전부 삭제
    delete_all_data()
    
    # category에 해당하는 폴더 찾기
    category =  request.form.get('client')
    dir = os.path.dirname(os.path.abspath(__file__))
    dir = os.path.join(dir, 'data')
    upload_dir = os.path.join(dir, category)
    files = os.listdir(upload_dir)

    #category에 해당하는 modelname을 불러옴
    model_name = select_from_model(category)
    if model_name and model_name[0]:
        model_name = model_name[0][0]
    else:
        model_name = None
    session['model_name'] = model_name
    mime_type =''
    for item in files :
        filepath = upload_dir + '/' + item
        filename = os.path.basename(filepath)

        if not os.path.exists(filepath):
            print('File does not exist.')
        else:
            if filename.endswith('.pdf'):
                mime_type = 'application/pdf'
            elif filename.endswith('.csv'):
                mime_type = 'text/csv'
            else:
                print('Unsupported file type')
                mime_type = None

            if mime_type is not None:
                with open(filepath, 'rb') as f:
                    file = f.read()

        url = 'http://127.0.0.1:8000/upsert-file'

        headers = {
            'accept': 'application/json',
            'Authorization': set_bearer_token
        }

        files = {
            'file': (filename, file, mime_type)
        }

        data = {
            'metadata': 'your metadata here'
        }
        requests.post(url, headers=headers, files=files, data=data)

    return render_template('index.html')

# 입력한 값 확인
@app.route('/get_data', methods=['POST'])
def get_data():
    url = 'http://127.0.0.1:8000/get_data'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': set_bearer_token
    }
    data = {}

    response = requests.post(url, headers=headers, json=data)
    files = response.json()

    return render_template('manage.html', files=files)

# 값 삭제
@app.route('/delete_data', methods=['POST'])
def delete_data():
# 1. 폴더 내 데이터 삭제
    file_name = request.form.get('file_name')
    file_URL = request.form.get('file_URL')
    file_URL = file_URL + '/' + file_name
    file_category = request.form.get('file_category')
    os.remove(file_URL)
# 2. 데이터베이스 내 데이터 삭제
    delete_from_db(file_name, file_category)
    return  render_template('manage.html')

# GPT 통신
@app.route('/action_page', methods=['POST'])
def action_page():
    model_name = session.get('model_name', None)
    print(model_name)

    MAX_INPUT_SIZE = 4096
    NUM_OUTPUTS = 1000
    MAX_CHUNK_OVERLAP = 0.2
    CHUNK_SIZE_LIMIT = 1000

    prompt_helper = PromptHelper(
        MAX_INPUT_SIZE, NUM_OUTPUTS, MAX_CHUNK_OVERLAP, chunk_size_limit=CHUNK_SIZE_LIMIT
    )

    llm_predictor = LLMPredictor(
        llm=OpenAI(
            temperature=0, model_name=model_name, max_tokens=1000
        )
    )
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )
    if 'chat_list' not in session:
        session['chat_list'] = []

    first_name = request.form.get('fname')
    if first_name.strip() != '':  # 사용자의 입력이 비어있지 않은 경우에만 추가

        reader = ChatGPTRetrievalPluginReader(
            endpoint_url="http://127.0.0.1:8000",
            bearer_token=bearer_token
        )
        documents = reader.load_data(first_name)
        index = ListIndex.from_documents(
            documents=documents, service_context=service_context
        )
        query_engine = index.as_query_engine(response_mode="compact")
        response = query_engine.query(first_name)
        response = str(response)
        bot_s = response.replace("\n", "<br>") # 챗 GPT 출력과 동일히 작성하기 위해 개행문자 변환
        user_s = first_name.replace("\n", "<br>")
        session['chat_list'].append({"user": user_s})  # 사용자 대화 세션 저장
        session['chat_list'].append({"bot": bot_s})  # 답변 대화 세션 저장 입력
        session.modified = True
    return render_template('index.html', dialogue=session['chat_list'], check_model=model_name)

# 파일 업로드 ** upsert 아님
@app.route('/send_upload', methods=['POST'])
def send_upload():
    category =  request.form.get('sports')
    dir = os.path.dirname(os.path.abspath(__file__))
    dir = os.path.join(dir, 'data')
    upload_dir = os.path.join(dir, category)

    # Create the upload directory if it doesn't exist
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    print(os.path.dirname(os.path.abspath(__file__)))
    print(upload_dir)
    print(type(upload_dir))

    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']

    if file.filename == '':
        return 'No selected file'
    if file:
        # Check file extension to set the correct MIME type
        if file.filename.endswith('.pdf'):
            mime_type = 'application/pdf'
        elif file.filename.endswith('.csv'):
            mime_type = 'text/csv'
        else:
            return 'Unsupported file type'

    file_path = os.path.join(upload_dir, file.filename)
    if os.path.exists(file_path):
        return 'File already exists in the directory'

    current_time = datetime.now()
    time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")

    file.save(os.path.join(upload_dir, file.filename))
    field_values = (file.filename, upload_dir, time_string, category)  # 삽입할 값
    insert_into_table(field_values)

    return render_template('manage.html', upload_response='Save data to path ' + upload_dir)

@app.route('/reset_session', methods=['POST'])
def reset_session():
    session['chat_list'] = []
    session.modified = True
    return render_template('index.html')

@app.route('/select_model', methods=['POST'])
def select_model():
    category =  request.form.get('model_category')
    model = request.form.get('model_type')
    field_values = (model, category)  # 삽입할 값
    insert_into_table(field_values)
    return render_template('manage.html')

def insert_into_table(field_values):
    db = set_db()
    try:
        if len(field_values) == 4:
            with db.cursor() as cursor:
                sql = "INSERT INTO listdata (filename, url, updatetime, category) VALUES (%s, %s, %s, %s)"
                cursor.execute(sql, field_values)
                db.commit()
        if len(field_values) == 2:
            with db.cursor() as cursor:
                sql = """
                INSERT INTO model (modelname, category) 
                VALUES (%s, %s)
                ON DUPLICATE KEY UPDATE 
                modelname = VALUES(modelname), category = VALUES(category)
                """
                cursor.execute(sql, field_values)
                db.commit()
    finally:
        db.close()  # 데이터베이스 연결 닫기
def set_db():
    db = pymysql.connect(
        host='localhost',
        user='root',
        password='1234!',
        db='sgsdb',
        charset='utf8',
    )
    return db

def select_from_table(category_value):
    db = set_db()
    result = []  # Initialize result
    try:
        with db.cursor() as cursor:
            if category_value == 'all':
                sql = "SELECT * FROM listdata"
                cursor.execute(sql)
            else:
                sql = "SELECT * FROM listdata WHERE category = %s"
                cursor.execute(sql, (category_value,))
            result = cursor.fetchall()
            print(result)
    finally:
        db.close()
    return result

def select_from_model(category_value):
    db = set_db()
    try:
        with db.cursor() as cursor:
            # SQL 쿼리에서 %s를 사용하여 인자를 넣을 위치를 표시
            sql = "SELECT modelname FROM model WHERE category = %s"
            cursor.execute(sql, (category_value,))
            results = cursor.fetchall()
    finally:
        db.close()

    # 결과 반환
    return results

def delete_from_db(name_value, category_value):
    db = set_db()
    try:
        with db.cursor() as cursor:
            sql = "DELETE FROM listdata WHERE filename = %s AND category = %s"
            cursor.execute(sql, (name_value, category_value))
            db.commit()
    finally:
        db.close()  # 데이터베이스 연결 닫기

def delete_all_data():
    url = 'http://127.0.0.1:8000/delete'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': set_bearer_token
    }
    data = {
        "delete_all": True
    }
    requests.delete(url, headers=headers, json=data)

if __name__ == '__main__':
    app.run(port=5050)