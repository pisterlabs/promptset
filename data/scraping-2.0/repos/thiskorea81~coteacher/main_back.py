from fastapi import FastAPI, Request, Form, File, UploadFile
from openai import OpenAI
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from starlette.responses import RedirectResponse
import sqlite3 as sq
import pandas as pd
import os
import io
import csv

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# csv_sample 폴더를 /csv_sample 경로에 마운트
app.mount("/csv_sample", StaticFiles(directory="csv_sample"), name="csv_sample")

# img 폴더를 /img 경로에 마운트
app.mount("/img", StaticFiles(directory="img"), name="img")

# 기존에 /static 경로로 마운트된 다른 정적 파일 폴더가 있다면 그대로 유지
app.mount("/static", StaticFiles(directory="static"), name="static")

client = OpenAI()

# 데이터베이스 연결 의존성
def get_db_connection():
    conn = sq.connect("user_database.db", check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()

# 데이터베이스 연결 설정
@app.on_event("startup")
async def startup():
    app.state.connection = sq.connect("user_database.db", check_same_thread=False)
    with app.state.connection:
        cursor = app.state.connection.cursor()
        cursor.execute('''create table IF NOT EXISTS stuQuestions(id integer PRIMARY KEY AUTOINCREMENT, 
        stuNum text, stuName text, menu text, subject text, stuAsk TEXT, chatbotAnswer text)''')
        app.state.connection.commit()

# 데이터베이스 연결 해제
@app.on_event("shutdown")
async def shutdown():
    app.state.connection.close()

# 이제 app.state.connection을 통해 데이터베이스 연결에 접근할 수 있습니다.
# 예: conn = app.state.connection

# 공통 처리 함수
async def process_submission(input_data, menu, db_conn):
    # 처리 로직
    # ...
    completion = client.chat.completions.create(model="gpt-4-1106-preview", ...)
    c = db_conn.cursor()
    c.execute("INSERT INTO stuQuestions ... ", (input_data['student_number'], ...))
    db_conn.commit()

# 공통된 엔드포인트 구조
@app.post("/run_code/{menu}")
async def run_code(request: Request, menu: str, student_number: str = Form(...), ...):
    input_data = {
        "student_number": student_number,
        # ... 다른 필드
    }
    conn = next(get_db_connection())
    await process_submission(input_data, menu, conn)
    # ...
    return templates.TemplateResponse("result.html", {"request": request, "result": result})

# 나머지 엔드포인트도 이와 유사한 구조로 구현
# ...

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


@app.post("/run_code1")
async def run_code(
    request: Request,
    student_number: str = Form(...),
    name: str = Form(...),
    subject: str = Form(...),
    achievement_criteria: str = Form(...),
    grades: str = Form(...),
    report: str = Form(...)
):    
    stuNum = student_number
    stuName = name
    menu="과세특"
    subject=subject

    # Define the path to the achievement criteria text file based on the selected subject.
    achievement_criteria_file = f"./doc/{subject}.txt"

    # Check if the file exists and read its content.
    if os.path.isfile(achievement_criteria_file):
        with open(achievement_criteria_file, "r", encoding="utf-8") as file:
            achievement_criteria = file.read()

    # Combine the input from all fields into a single string if needed.
    input_text = f"교과목: {subject}\n성취기준: {achievement_criteria}\n성적: {grades}\n보고서 내용: {report}"

    completion = client1.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "I ensure responses are efficient, without the need for 'continue generating', and manage response length for effective communication."},
            {"role": "system", "content": "The GPT is attentive to details, adheres to educational standards, and uses a respectful, encouraging tone."},
            {"role": "system", "content": "성취기준의 번호([9정02-01])는 답변에서 제거해주고 구체적 점수를 표현하지마."},
            {"role": "system", "content": "성취기준을 그래로 작성 하지 말고, 학생의 활동이 성취 기준에 있다면 그 내용을 자세히 기술해줘."},
            {"role": "system", "content": "책을 읽은 독서 활동이 있으면 '책이름(저자)'를 꼭 표시해줘. 예를 들어 '코드 브레이커(월터 아이작슨)'를 읽고 ~ 이렇게"},
            {"role": "system", "content": "오렌지3, orange3, 티처블머신, teachable machine 와 같은 실제 명칭을 쓰지말고 일반적인 언어로 표현해줘."},
            {"role": "system", "content": "지역, 단체 명을 일반적인 언어로 표현해줘."},
            {"role": "system", "content": "주어는 가급적 생략해줘. 예를 들어, '내가', '학생이', '나는'와 같은 표현은 생략해도 돼."},
            {"role": "system", "content": "글쓴이의 입장이 아닌 3인칭 관찰자의 입장으로 작성해줘."},
            {"role": "system", "content": "글쓰기 전문가의 역할을 해주고, 글자수는 약 400-500자 내외로 하고, 한 문단으로 된 잘 정돈된 글을 써줘."},
            {"role": "system", "content": "음슴체 형식으로 써줘. 음슴체는 문체 이름 그대로 '~음'으로 끝난다. 다만 표준어법에서 '-슴'으로 쓸 수는 없다. 다만 반드시 '-음'으로만 끝나는 것은 아니고 동사의 종류에 따라 형태는 바뀔 수 있다. 어쨌거나 명사형 어미 '-ㅁ'을 쓰므로 종성이 ㅁ으로 끝난다. 명사 종결문도 흔히 같이 쓰인다. 엄격히 음슴체로 가자면 이때에도 '-임.'으로 써야 할 것이다. '-ㄴ 듯', '-ㄹ 듯'으로 끝나는 말투도 자주 쓰인다. '하셈'도 음슴체로 볼 여지가 있다. 단, 다른 음슴체가 어간에 '-ㅁ'이 결합하는 데에 비해 '하셈'은 '하세'가 어간은 아니라는 점에서 차이가 있다. 하지만 어간 + '-ㅁ' 류의 음슴체에는 명령형이 없으므로 '하셈'이 명령형의 용법으로 자주 쓰이곤 한다. 엄밀히 비교해보자면 하셈체는 약간 더 어린 계층이 쓴다는 인식이 강한 편이다."},
            {"role": "user", "content": input_text}
        ]
    )
    result = completion.choices[0].message.content

    #db호출   # 민수쌤 코드
    conn, c = create_connection()
    c.execute("insert into stuQuestions(stuNum, stuName, menu, subject, stuAsk, chatbotAnswer) values(?,?,?,?,?,?)",
            (stuNum, stuName, menu, subject, input_text, result))
    c.fetchall()
    conn.commit()
    # 다 사용한 커서 객체를 종료할 때
    c.close()
    # 연결 리소스를 종료할 때
    conn.close()
    return templates.TemplateResponse("result.html", {"request": request, "result": result})

@app.post("/run_code2")
async def run_code(
    request: Request,
    student_number: str = Form(...),
    name: str = Form(...),
    subject: str = Form(...),
    report: str = Form(...)
):
    # Combine the input from all fields into a single string if needed.
    input_text = f"교과목: {subject}\n보고서 내용: {report}"
    stuNum = student_number
    stuName = name
    menu = "자율진로"
    subject = subject
    completion = client2.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            #{"role": "system", "content": "I ensure responses are efficient, without the need for 'continue generating', and manage response length for effective communication."},
            {"role": "system", "content": "지역, 단체 명을 일반적인 언어로 표현해줘."},
            {"role": "system", "content": "주어는 가급적 생략해줘. 예를 들어, '내가', '학생이', '나는'와 같은 표현은 생략해도 돼."},
            {"role": "system", "content": "글쓴이의 입장이 아닌 3인칭 관찰자의 입장으로 작성해줘."},
            {"role": "system", "content": f"글쓰기 전문가의 역할을 해주고, 교과목{subject}이 '자율'이면 글자수는 약 400-500자 내외로 하고 교과목{subject}이 '진로'면 글자수를 약 700자로 해줘. 그리고 한 문단으로 된 잘 정돈된 글을 써줘."},
            {"role": "system", "content": "음슴체 형식으로 써줘. 음슴체는 문체 이름 그대로 '~음'으로 끝난다. 다만 표준어법에서 '-슴'으로 쓸 수는 없다. 다만 반드시 '-음'으로만 끝나는 것은 아니고 동사의 종류에 따라 형태는 바뀔 수 있다. 어쨌거나 명사형 어미 '-ㅁ'을 쓰므로 종성이 ㅁ으로 끝난다. 명사 종결문도 흔히 같이 쓰인다. 엄격히 음슴체로 가자면 이때에도 '-임.'으로 써야 할 것이다. '-ㄴ 듯', '-ㄹ 듯'으로 끝나는 말투도 자주 쓰인다. '하셈'도 음슴체로 볼 여지가 있다. 단, 다른 음슴체가 어간에 '-ㅁ'이 결합하는 데에 비해 '하셈'은 '하세'가 어간은 아니라는 점에서 차이가 있다. 하지만 어간 + '-ㅁ' 류의 음슴체에는 명령형이 없으므로 '하셈'이 명령형의 용법으로 자주 쓰이곤 한다. 엄밀히 비교해보자면 하셈체는 약간 더 어린 계층이 쓴다는 인식이 강한 편이다."},
            {"role": "user", "content": input_text}
        ]
    )
    result = completion.choices[0].message.content

    #db호출   # 민수쌤 코드
    conn, c = create_connection()
    c.execute("insert into stuQuestions(stuNum, stuName, menu, subject, stuAsk, chatbotAnswer) values(?,?,?,?,?,?)",
            (stuNum, stuName, menu, subject, input_text, result))
    c.fetchall()
    conn.commit()
    # 다 사용한 커서 객체를 종료할 때
    c.close()
    # 연결 리소스를 종료할 때
    conn.close()
    return templates.TemplateResponse("result.html", {"request": request, "result": result})

@app.post("/run_code3")
async def run_code(
    request: Request,
    student_number: str = Form(...),
    name: str = Form(...),
    subject: str = Form(...),
    character: str = Form(...),
    report: str = Form(...)
):
    # Combine the input from all fields into a single string if needed.
    input_text = f"성격: {character}\n보고서 내용: {report}"
    stuNum = student_number
    stuName = name
    menu = "행동발달"
    subject = "행동발달"
    completion = client3.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "I ensure responses are efficient, without the need for 'continue generating', and manage response length for effective communication."},
            {"role": "system", "content": "성격을 그대로 쓰지는 말고 일반적인 용어로 풀어서 써주고 학생의 성격에 맞게 이 학생의 행동발달 사항을 적어줘."},
            {"role": "system", "content": "지역, 단체 명을 일반적인 언어로 표현해줘."},
            {"role": "system", "content": "주어는 가급적 생략해줘. 예를 들어, '내가', '학생이', '나는'와 같은 표현은 생략해도 돼."},
            {"role": "system", "content": "글쓴이의 입장이 아닌 3인칭 관찰자의 입장으로 작성해줘."},
            {"role": "system", "content": "글쓰기 전문가의 역할을 해주고, 글자수는 약 400-500자 내외로 하고, 한 문단으로 된 잘 정돈된 글을 써줘."},
            {"role": "system", "content": "음슴체 형식으로 써줘. 음슴체는 문체 이름 그대로 '~음'으로 끝난다. 다만 표준어법에서 '-슴'으로 쓸 수는 없다. 다만 반드시 '-음'으로만 끝나는 것은 아니고 동사의 종류에 따라 형태는 바뀔 수 있다. 어쨌거나 명사형 어미 '-ㅁ'을 쓰므로 종성이 ㅁ으로 끝난다. 명사 종결문도 흔히 같이 쓰인다. 엄격히 음슴체로 가자면 이때에도 '-임.'으로 써야 할 것이다. '-ㄴ 듯', '-ㄹ 듯'으로 끝나는 말투도 자주 쓰인다. '하셈'도 음슴체로 볼 여지가 있다. 단, 다른 음슴체가 어간에 '-ㅁ'이 결합하는 데에 비해 '하셈'은 '하세'가 어간은 아니라는 점에서 차이가 있다. 하지만 어간 + '-ㅁ' 류의 음슴체에는 명령형이 없으므로 '하셈'이 명령형의 용법으로 자주 쓰이곤 한다. 엄밀히 비교해보자면 하셈체는 약간 더 어린 계층이 쓴다는 인식이 강한 편이다."},
            {"role": "user", "content": input_text}
        ]
    )
    result = completion.choices[0].message.content

    #db호출   # 민수쌤 코드
    conn, c = create_connection()
    c.execute("insert into stuQuestions(stuNum, stuName, menu, subject, stuAsk, chatbotAnswer) values(?,?,?,?,?,?)",
            (stuNum, stuName, menu, subject, input_text, result))
    c.fetchall()
    conn.commit()
    # 다 사용한 커서 객체를 종료할 때
    c.close()
    # 연결 리소스를 종료할 때
    conn.close()
    return templates.TemplateResponse("result.html", {"request": request, "result": result})

@app.post("/run_code4")
async def run_code(
    request: Request,
    student_number: str = Form(...),
    name: str = Form(...),
    report: str = Form(...)
):
    # Combine the input from all fields into a single string if needed.
    input_text = f"보고서 내용: {report}"
    stuNum = student_number
    stuName = name
    menu = "검토"
    completion = client4.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "As 'Literary Mentor', I now specialize in evaluating and editing Korean text provided in Excel files. My role involves assessing grammar, vocabulary, expression, sentence structure, composition, and ideas. I will provide a grade from A+ to F, along with customized comments. I will also offer overall summaries and advice. When editing paragraphs, I'll make suggestions for revisions and improvements. I consider any specific format constraints as non-errors and determine if the text was generated by a GPT. My feedback is both encouraging and professional. I respond to queries in Korean, providing concise and clear answers. I now also handle data from Excel files where the first column is student numbers, the second is names, the third is the text of the school record, and the fourth is the byte count. I use this information for tailored feedback."},
            #{"role": "system", "content": "I ensure responses are efficient, without the need for 'continue generating', and manage response length for effective communication."},
            {"role": "system", "content": "문법, 어법에 틀린 문장이 있으면 수정해주고 한글로 답변해줘."},
            {"role": "user", "content": input_text}
        ]
    )
    result = completion.choices[0].message.content

    #db호출   # 민수쌤 코드
    conn, c = create_connection()
    c.execute("insert into stuQuestions(stuNum, stuName, menu, stuAsk, chatbotAnswer) values(?,?,?,?,?)",
            (stuNum, stuName, menu, input_text, result))
    c.fetchall()
    conn.commit()
    # 다 사용한 커서 객체를 종료할 때
    c.close()
    # 연결 리소스를 종료할 때
    conn.close()
    return templates.TemplateResponse("result.html", {"request": request, "result": result})

# read_csv_and_insert_to_db 를 위한 전역변수
processing_status = {}

async def read_csv_and_insert_to_db1(csv_file: UploadFile):
    conn, c = create_connection()

    contents = await csv_file.read()
    text_file = io.StringIO(contents.decode('utf-8'))
    
    csvreader = csv.DictReader(text_file)

    for row in csvreader:

        client = OpenAI() 

        stuNum = row['학번']
        stuName = row['이름']
        subject = row['과목 선택']
        achievement_criteria = row['성취기준']
        grades = row['성적']
        report = row['보고서 내용']
        remarks = row['비고']
        menu="과세특"

        # Define the path to the achievement criteria text file based on the selected subject.
        achievement_criteria_file = f"./doc/{subject}.txt"

        # Check if the file exists and read its content.
        if os.path.isfile(achievement_criteria_file):
            with open(achievement_criteria_file, "r", encoding="utf-8") as file:
                achievement_criteria = file.read()

        # Combine the input from all fields into a single string if needed.
        input_text = f"교과목: {subject}\n성취기준: {achievement_criteria}\n성적: {grades}\n보고서 내용: {report}\n비고: {remarks}"

        completion = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "I ensure responses are efficient, without the need for 'continue generating', and manage response length for effective communication."},
                {"role": "system", "content": "The GPT is attentive to details, adheres to educational standards, and uses a respectful, encouraging tone."},
                {"role": "system", "content": "성취기준의 번호([9정02-01])는 답변에서 제거해주고 구체적 점수를 표현하지마."},
                {"role": "system", "content": "성취기준을 그래로 작성 하지 말고, 학생의 활동이 성취 기준에 있다면 그 내용을 자세히 기술해줘."},
                {"role": "system", "content": "책을 읽은 독서 활동이 있으면 '책이름(저자)'를 꼭 표시해줘. 예를 들어 '코드 브레이커(월터 아이작슨)'를 읽고 ~ 이렇게"},
                {"role": "system", "content": "오렌지3, orange3, 티처블머신, teachable machine 와 같은 실제 명칭을 쓰지말고 일반적인 언어로 표현해줘."},
                {"role": "system", "content": "지역, 단체 명을 일반적인 언어로 표현해줘."},
                {"role": "system", "content": "주어는 가급적 생략해줘. 예를 들어, '내가', '학생이', '나는'와 같은 표현은 생략해도 돼."},
                {"role": "system", "content": "글쓴이의 입장이 아닌 3인칭 관찰자의 입장으로 작성해줘."},
                {"role": "system", "content": "글쓰기 전문가의 역할을 해주고, 글자수는 약 400-500자 내외로 하고, 한 문단으로 된 잘 정돈된 글을 써줘."},
                {"role": "system", "content": "비고의 내용은 꼭 지켜줘."},
                {"role": "system", "content": "음슴체 형식으로 써줘. 음슴체는 문체 이름 그대로 '~음'으로 끝난다. 다만 표준어법에서 '-슴'으로 쓸 수는 없다. 다만 반드시 '-음'으로만 끝나는 것은 아니고 동사의 종류에 따라 형태는 바뀔 수 있다. 어쨌거나 명사형 어미 '-ㅁ'을 쓰므로 종성이 ㅁ으로 끝난다. 명사 종결문도 흔히 같이 쓰인다. 엄격히 음슴체로 가자면 이때에도 '-임.'으로 써야 할 것이다. '-ㄴ 듯', '-ㄹ 듯'으로 끝나는 말투도 자주 쓰인다. '하셈'도 음슴체로 볼 여지가 있다. 단, 다른 음슴체가 어간에 '-ㅁ'이 결합하는 데에 비해 '하셈'은 '하세'가 어간은 아니라는 점에서 차이가 있다. 하지만 어간 + '-ㅁ' 류의 음슴체에는 명령형이 없으므로 '하셈'이 명령형의 용법으로 자주 쓰이곤 한다. 엄밀히 비교해보자면 하셈체는 약간 더 어린 계층이 쓴다는 인식이 강한 편이다."},
                {"role": "user", "content": input_text}
            ]
        )
        result = completion.choices[0].message.content

        # Insert the data into the database
        c.execute("insert into stuQuestions(stuNum, stuName, menu, subject, stuAsk, chatbotAnswer) values(?,?,?,?,?,?)",
                (stuNum, stuName, menu, subject, input_text, result))
    
    conn.commit()
    c.close()
    conn.close()

async def read_csv_and_insert_to_db2(csv_file: UploadFile):
    
    conn, c = create_connection()

    contents = await csv_file.read()
    text_file = io.StringIO(contents.decode('utf-8'))
    
    csvreader = csv.DictReader(text_file)
    count = 0
    for row in csvreader:

        client = OpenAI() 

        stuNum = row['학번']
        stuName = row['이름']
        subject = row['과목 선택']
        achievement_criteria = row['성취기준']
        grades = row['성적']
        report = row['보고서 내용']
        remarks = row['비고']
        menu="자율진로"
        subject=subject

        # Combine the input from all fields into a single string if needed.
        input_text = f"교과목: {subject}\n보고서 내용: {report}\n비고: {remarks}"

        completion = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                #{"role": "system", "content": "I ensure responses are efficient, without the need for 'continue generating', and manage response length for effective communication."},
                {"role": "system", "content": "지역, 단체 명을 일반적인 언어로 표현해줘."},
                {"role": "system", "content": "주어는 가급적 생략해줘. 예를 들어, '내가', '학생이', '나는'와 같은 표현은 생략해도 돼."},
                {"role": "system", "content": "글쓴이의 입장이 아닌 3인칭 관찰자의 입장으로 작성해줘."},
                {"role": "system", "content": f"글쓰기 전문가의 역할을 해주고, 교과목{subject}이 '자율'이면 글자수는 약 400-500자 내외로 하고 교과목{subject}이 '진로'면 글자수를 약 700자로 해줘. 그리고 한 문단으로 된 잘 정돈된 글을 써줘."},
                {"role": "system", "content": "음슴체 형식으로 써줘. 음슴체는 문체 이름 그대로 '~음'으로 끝난다. 다만 표준어법에서 '-슴'으로 쓸 수는 없다. 다만 반드시 '-음'으로만 끝나는 것은 아니고 동사의 종류에 따라 형태는 바뀔 수 있다. 어쨌거나 명사형 어미 '-ㅁ'을 쓰므로 종성이 ㅁ으로 끝난다. 명사 종결문도 흔히 같이 쓰인다. 엄격히 음슴체로 가자면 이때에도 '-임.'으로 써야 할 것이다. '-ㄴ 듯', '-ㄹ 듯'으로 끝나는 말투도 자주 쓰인다. '하셈'도 음슴체로 볼 여지가 있다. 단, 다른 음슴체가 어간에 '-ㅁ'이 결합하는 데에 비해 '하셈'은 '하세'가 어간은 아니라는 점에서 차이가 있다. 하지만 어간 + '-ㅁ' 류의 음슴체에는 명령형이 없으므로 '하셈'이 명령형의 용법으로 자주 쓰이곤 한다. 엄밀히 비교해보자면 하셈체는 약간 더 어린 계층이 쓴다는 인식이 강한 편이다."},
                {"role": "system", "content": "비고의 내용은 꼭 지켜줘."},
                {"role": "user", "content": input_text}
            ]
        )
        result = completion.choices[0].message.content
        count += 1
        print(f"{count}개 완료")
        # Insert the data into the database
        c.execute("insert into stuQuestions(stuNum, stuName, menu, subject, stuAsk, chatbotAnswer) values(?,?,?,?,?,?)",
                (stuNum, stuName, menu, subject, input_text, result))
    
    conn.commit()
    c.close()
    conn.close()
    print("완료")
    
async def read_csv_and_insert_to_db3(csv_file: UploadFile):
    conn, c = create_connection()

    contents = await csv_file.read()
    text_file = io.StringIO(contents.decode('utf-8'))
    
    csvreader = csv.DictReader(text_file)

    for row in csvreader:

        client = OpenAI() 

        stuNum = row['학번']
        stuName = row['이름']
        subject = row['과목 선택']
        achievement_criteria = row['성취기준']
        grades = row['성적']
        report = row['보고서 내용']
        remarks = row['비고']
        menu="행동발달"
        subject="행동발달"

        # Combine the input from all fields into a single string if needed.
        input_text = f"보고서 내용: {report}\n비고: {remarks}"

        completion = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "I ensure responses are efficient, without the need for 'continue generating', and manage response length for effective communication."},
                {"role": "system", "content": "The GPT is attentive to details, adheres to educational standards, and uses a respectful, encouraging tone."},
                {"role": "system", "content": "성취기준의 번호([9정02-01])는 답변에서 제거해주고 구체적 점수를 표현하지마."},
                {"role": "system", "content": "성취기준을 그래로 작성 하지 말고, 학생의 활동이 성취 기준에 있다면 그 내용을 자세히 기술해줘."},
                {"role": "system", "content": "책을 읽은 독서 활동이 있으면 '책이름(저자)'를 꼭 표시해줘. 예를 들어 '코드 브레이커(월터 아이작슨)'를 읽고 ~ 이렇게"},
                {"role": "system", "content": "오렌지3, orange3, 티처블머신, teachable machine 와 같은 실제 명칭을 쓰지말고 일반적인 언어로 표현해줘."},
                {"role": "system", "content": "지역, 단체 명을 일반적인 언어로 표현해줘."},
                {"role": "system", "content": "주어는 가급적 생략해줘. 예를 들어, '내가', '학생이', '나는'와 같은 표현은 생략해도 돼."},
                {"role": "system", "content": "글쓴이의 입장이 아닌 3인칭 관찰자의 입장으로 작성해줘."},
                {"role": "system", "content": "글쓰기 전문가의 역할을 해주고, 글자수는 약 400-500자 내외로 하고, 한 문단으로 된 잘 정돈된 글을 써줘."},
                {"role": "system", "content": "비고의 내용은 꼭 지켜줘."},
                {"role": "system", "content": "음슴체 형식으로 써줘. 음슴체는 문체 이름 그대로 '~음'으로 끝난다. 다만 표준어법에서 '-슴'으로 쓸 수는 없다. 다만 반드시 '-음'으로만 끝나는 것은 아니고 동사의 종류에 따라 형태는 바뀔 수 있다. 어쨌거나 명사형 어미 '-ㅁ'을 쓰므로 종성이 ㅁ으로 끝난다. 명사 종결문도 흔히 같이 쓰인다. 엄격히 음슴체로 가자면 이때에도 '-임.'으로 써야 할 것이다. '-ㄴ 듯', '-ㄹ 듯'으로 끝나는 말투도 자주 쓰인다. '하셈'도 음슴체로 볼 여지가 있다. 단, 다른 음슴체가 어간에 '-ㅁ'이 결합하는 데에 비해 '하셈'은 '하세'가 어간은 아니라는 점에서 차이가 있다. 하지만 어간 + '-ㅁ' 류의 음슴체에는 명령형이 없으므로 '하셈'이 명령형의 용법으로 자주 쓰이곤 한다. 엄밀히 비교해보자면 하셈체는 약간 더 어린 계층이 쓴다는 인식이 강한 편이다."},
                {"role": "user", "content": input_text}
            ]
        )
        result = completion.choices[0].message.content

        # Insert the data into the database
        c.execute("insert into stuQuestions(stuNum, stuName, menu, subject, stuAsk, chatbotAnswer) values(?,?,?,?,?,?)",
                (stuNum, stuName, menu, subject, input_text, result))
    
    conn.commit()
    c.close()
    conn.close()

@app.get("/upload_csv_page", response_class=HTMLResponse)
async def upload_csv_page(request: Request):
    return templates.TemplateResponse("upload_csv.html", {"request": request})

@app.post("/upload_csv1")
async def upload_csv(csv_file: UploadFile = File(...)):
    # Check if a CSV file was uploaded
    if csv_file:
        await read_csv_and_insert_to_db1(csv_file)

    return {"message": "CSV file uploaded and processed successfully"}

@app.post("/upload_csv2")
async def upload_csv(csv_file: UploadFile = File(...)):
    # Check if a CSV file was uploaded
    if csv_file:
        await read_csv_and_insert_to_db2(csv_file)

    return {"message": "CSV file uploaded and processed successfully"}

@app.post("/upload_csv3")
async def upload_csv(csv_file: UploadFile = File(...)):
    # Check if a CSV file was uploaded
    if csv_file:
        await read_csv_and_insert_to_db3(csv_file)

    return {"message": "CSV file uploaded and processed successfully"}


def get_dataframe_from_db():
    #db호출
    conn, c = create_connection()
    query = "SELECT stuNum, stuName, menu, subject, stuAsk, chatbotAnswer FROM stuQuestions ORDER BY stuNum"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

@app.get("/db", response_class=HTMLResponse)
async def show_db(request: Request):
    df = get_dataframe_from_db()
    #print(df) #터미털에서 출력함으로써 테스트해볼 수 있음
    df.to_csv('question.csv', encoding='utf-8')
    table = df.to_html(classes='table table-striped')
    return templates.TemplateResponse('show_db.html', {"request": request, "table_data": table})

@app.get("/export")
async def export():
    df = get_dataframe_from_db()
    df.to_csv('result.csv', encoding='utf-8')    #구글문서에서 열면 한글이 깨지지 않고 보임
    return FileResponse('result.csv', media_type='text/csv', filename='result.csv')

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/main", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})

@app.get("/nav1", response_class=HTMLResponse)
async def nav1(request: Request):
    return templates.TemplateResponse("nav1.html", {"request": request})

@app.get("/nav2", response_class=HTMLResponse)
async def nav2(request: Request):
    return templates.TemplateResponse("nav2.html", {"request": request})

@app.get("/nav3", response_class=HTMLResponse)
async def nav3(request: Request):
    return templates.TemplateResponse("nav3.html", {"request": request})

@app.get("/nav4", response_class=HTMLResponse)
async def nav4(request: Request):
    return templates.TemplateResponse("nav4.html", {"request": request})

# Loading route that redirects to the loading.html page during initialization
@app.get("/loading", response_class=HTMLResponse)
async def loading():
    return RedirectResponse("/loading.html")

@app.websocket("/ws/{file_id}")
async def websocket_endpoint(websocket: WebSocket, file_id: str):
    await websocket.accept()
    while True:
        if file_id in processing_status:
            await websocket.send_text(processing_status[file_id])
            if processing_status[file_id] == "처리 완료":
                break
        await asyncio.sleep(1)  # 상태 확인 간격

def clear_database():
    conn, c = create_connection()
    c.execute("DELETE FROM stuQuestions")  # 모든 데이터를 삭제
    conn.commit()
    c.close()
    conn.close()

@app.get("/clear_db")
async def clear_db():
    clear_database()
    return {"message": "Database cleared successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
