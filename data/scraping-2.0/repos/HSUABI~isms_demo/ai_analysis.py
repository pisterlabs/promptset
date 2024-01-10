import os
from openai import OpenAI
import PyPDF2
from konlpy.tag import Okt
import re
import paramiko
import ntplib
from time import ctime
import datetime

def servercheck():
    # SSH 접속 정보
    hostname = '192.168.230.133'
    username = 'qwer'
    password = '1234'

    # 반환할 결과 문자열
    result = ""

    # SSH 클라이언트 설정
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # 서버에 연결
        client.connect(hostname, username=username, password=password)

        # 1. admin 또는 administrator 계정 확인
        stdin, stdout, stderr = client.exec_command('cat /etc/passwd | egrep "admin:|administrator:"')
        admin_accounts = stdout.read().decode()
        result += '1. Admin/Administrator Accounts Check:\n'
        result += admin_accounts if admin_accounts else "No admin/administrator accounts found"
        result += '\n'

        # 2. 세션 타임아웃 체크

        stdin, stdout, stderr = client.exec_command('cat /etc/profile | grep "export TMOUT="')
        tmout_line = stdout.read().decode().strip()
        result += '2. Session Timeout Check:\n'
        if tmout_line:
            # TMOUT 값을 추출
            session_timeout = tmout_line.split('=')[1]
            result +=f"Session timeout is {session_timeout} seconds"
        else:
            result +="Session timeout not set or less than 300 seconds"
        result += '\n'
        

        # 3. 비밀번호 정책 설정값 확인
        stdin, stdout, stderr = client.exec_command('cat /etc/login.defs | egrep "PASS_MIN_DAYS|PASS_MAX_DAYS|PASS_MIN_LEN" | egrep -v "#"')
        password_policy = stdout.read().decode()
        result += '3. Password Policy Check:\n'
        result += password_policy
        result += '\n'

        # 4. 시간 동기화 확인 (Google NTP 시간 비교)
        ntp_client = ntplib.NTPClient()
        ntp_server = 'time.google.com'
        response = ntp_client.request(ntp_server, version=3)
        google_time = ctime(response.tx_time)
        stdin, stdout, stderr = client.exec_command('date +"%Y-%m-%d %H:%M:%S"')
        server_time_str = stdout.read().decode().strip()
        server_time = datetime.datetime.strptime(server_time_str, "%Y-%m-%d %H:%M:%S")
        google_time_dt = datetime.datetime.strptime(google_time, "%a %b %d %H:%M:%S %Y")
        time_diff = abs((google_time_dt - server_time).total_seconds())
        result += '4. Time Synchronization Check:\n'
        result += f"Server time: {server_time_str}, Google time: {google_time}\n"
        result += f"Time difference: {time_diff} seconds\n"
        result += "Time difference is more than 30 seconds.\n" if time_diff > 30 else "Time is synchronized within 30 seconds.\n"

        # 5. 백신 설치 여부 확인
        stdin, stdout, stderr = client.exec_command('find / -name "v3net"')
        antivirus_check = stdout.read().decode()
        result += '5. Antivirus(V3) Installation Check:\n'
        result += "Antivirus installed\n" if antivirus_check else "No antivirus installed(V3)\n"

    except Exception as e:
        result += f"Error: {e}\n"
    finally:
        # 연결 종료
        client.close()

    return result

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# 예제 사용
pdf_path = './sample.pdf'  # 여기에 PDF 파일 경로를 입력하세요
extracted_text = extract_text_from_pdf(pdf_path)

OPENAI_API_KEY = "sk-mEhJbVxFCAJkWj22OeJLT3BlbkFJAKY65v4N1qADAdgmAb77AAAAAAAA"

content = extracted_text

model = "gpt-4"

# openai.api_key = OPENAI_API_KEY

client = OpenAI(
    # This is the default and can be omitted
    api_key=OPENAI_API_KEY,
)

file_252 = open('result_2.5.2.txt', 'w')
file_253 = open('result_2.5.3.txt', 'w')
file_254 = open('result_2.5.4.txt', 'w')
file_296 = open('result_2.9.6.txt', 'w')
file_2109 = open('result_2.10.9.txt', 'w')

#관리자계정, 세션타임아웃, 패스워드 최대사용기간, 시간동기화, v3백신설치 내용은 제외해서 말해라
item = "계정 정책(계정 목록)"
standard = "관리자 및 특수권한 계정은 쉽게 추측 가능한 식별자(root, admin, administrator 등)의 사용을 제한"
server_result = servercheck()
stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": f"원래 규칙은 {standard}이다. 서버점검결과를 검토하려고한다.서버점검결과는 원래규칙과 같아야한다. 만약다르다면 원래규칙이 맞는것이고 서버점검결과가 틀린것이다. 서버의 실제 점검결과인 \n{server_result}\n 에서 {item} 설정 부분이 원래 규칙에 부합하는지 확인해라. 대답할때 원래규칙을 명시하고, 서버 실제점검결과도 명시해라. 세션타임아웃, 패스워드 최대사용기간, 시간동기화, v3백신설치 내용은 제외해서 말해라" }],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        a=chunk.choices[0].delta.content
        file_252.write(a)
        print(a, end="")

print("\n-----------------------------------------\n")

item = "세션타임아웃 시간"
standard = "세션타임아웃 시간 300초 미만"
server_result = servercheck()
stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": f"원래 규칙은 {standard}이다. 증빙자료와 서버점검결과를 검토하려고한다. 증빙자료와 서버점검결과는 원래규칙과 같아야한다. 만약다르다면 원래규칙이 맞는것이고 증빙자료와 서버점검결과가 틀린것이다. 증빙자료 \n{content}\n 에서 {item} 설정 부분과 서버의 실제 점검결과인 \n{server_result}\n 에서 {item} 설정 부분, 총 2개의 설정부분이 원래 규칙에 부합하는지 확인해라. 대답할때 원래규칙을 명시하고, 서버 실제점검결과와 증빙자료의 설정부분을 각각 따로 따로 명시해라. 관리자계정, 패스워드 최대사용기간, 시간동기화, v3백신설치 내용은 제외해서 말해라" }],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        a=chunk.choices[0].delta.content
        file_253.write(a)
        print(a, end="")

print("\n-----------------------------------------\n")

item = "패스워드 최대 사용기간"
standard = "패스워드 최대 사용기간 30일 이하"
server_result = servercheck()
stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": f"원래 규칙은 {standard}이다. 증빙자료와 서버점검결과를 검토하려고한다. 증빙자료와 서버점검결과는 원래규칙과 같아야한다. 만약다르다면 원래규칙이 맞는것이고 증빙자료와 서버점검결과가 틀린것이다. 증빙자료 \n{content}\n 에서 {item} 설정 부분과 서버의 실제 점검결과인 \n{server_result}\n 에서 {item} 설정 부분, 총 2개의 설정부분이 원래 규칙에 부합하는지 확인해라. 대답할때 원래규칙을 명시하고, 서버 실제점검결과와 증빙자료의 설정부분을 각각 따로 따로 명시해라. 관리자계정, 세션타임아웃, 시간동기화, v3백신설치 내용은 제외해서 말해라" }],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        a=chunk.choices[0].delta.content
        file_254.write(a)
        print(a, end="")


print("\n-----------------------------------------\n")

item = "시간동기화"
standard = "시간이 구글서버시간과 오차가 30초이내"
server_result = servercheck()
stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": f"원래 규칙은 {standard}이다. 증빙자료와 서버점검결과를 검토하려고한다. 증빙자료와 서버점검결과는 원래규칙과 같아야한다. 만약다르다면 원래규칙이 맞는것이고 증빙자료와 서버점검결과가 틀린것이다. 증빙자료 \n{content}\n 에서 {item} 설정 부분과 서버의 실제 점검결과인 \n{server_result}\n 에서 {item} 설정 부분, 총 2개의 설정부분이 원래 규칙에 부합하는지 확인해라. 대답할때 원래규칙을 명시하고, 서버 실제점검결과와 증빙자료의 설정부분을 각각 따로 따로 명시해라. 관리자계정, 세션타임아웃, 패스워드 최대사용기간, v3백신설치 내용은 제외해서 말해라" }],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        a=chunk.choices[0].delta.content
        file_296.write(a)
        print(a, end="")

print("\n-----------------------------------------\n")

item = "백신설치"
standard = "v3백신 설치"
server_result = servercheck()
stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": f"원래 규칙은 {standard}이다. 증빙자료와 서버점검결과를 검토하려고한다. 증빙자료와 서버점검결과는 원래규칙과 같아야한다. 만약다르다면 원래규칙이 맞는것이고 증빙자료와 서버점검결과가 틀린것이다. 증빙자료 \n{content}\n 에서 {item} 설정 부분과 서버의 실제 점검결과인 \n{server_result}\n 에서 {item} 설정 부분, 총 2개의 설정부분이 원래 규칙에 부합하는지 확인해라. 대답할때 원래규칙을 명시하고, 서버 실제점검결과와 증빙자료의 설정부분을 각각 따로 따로 명시해라. 관리자계정, 세션타임아웃, 패스워드 최대사용기간, 시간동기화 내용은 제외해서 말해라" }],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        a=chunk.choices[0].delta.content
        file_2109.write(a)
        print(a, end="")


        