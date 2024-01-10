import sys
sys.path.append(r'C:\ReviewService\package')

from dbconn import mysqlDbConnection, mysqlDbClose
import openai

key_file = open('C:\ReviewService\GptApiService\openai_apikey.txt','r', encoding='UTF8')
openai.api_key = key_file.readline()
key_file.close()

dbConn = mysqlDbConnection('root', '0000', '127.0.0.1', 3306, 'reviewdb')
cursor = dbConn.cursor()

def call_gptapi_type(data):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "The reviews I ask you are all negative, so please respond in Korean." }, # 역할지시문은 한국어보다 영어로 작성해야 더 잘 동작함
            {"role": "user", "content": f"{data} 이 후기 내용들이 '서비스불만', '배송불만', '제품불만'중에 무엇인지 5글자 이내로 답변해주세요."}
        ],
        timeout = 10    
        )
    return response

def call_gptapi_improvement(data):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Please provide specific suggestions for addressing the issues mentioned in this review. It's not the customer who sees your answer, it's the company employee. Respond in Korean." }, # 역할지시문은 한국어보다 영어로 작성해야 더 잘 동작함
            {"role": "user", "content":f"{data} 회사 직원의 입장에서 다음과 같은 고객의 부정적인 후기에 대해 어떤 개선 방안을 제안받을 수 있을까요? 300자 이내로 서술해주세요."}
        ],
            timeout = 10    
        )

    return response

def gpt_badreview(id):
    select_query = f"SELECT id, productName, title, content, gpt_answer FROM ttobak_review WHERE id='{id}' AND (gpt_answer = '나쁨' OR gpt_answer = '매우나쁨')"
    cursor.execute(select_query)

    data = cursor.fetchone()

    if not data:
        return
    
    type_response = None
    improvement_response = None

    try:    
        type_response = call_gptapi_type(data[3])
        improvement_response = call_gptapi_improvement(data[3])
    except:
        print("예외가 한번 발생")
        try:
            type_response = call_gptapi_type(data[3])
            improvement_response = call_gptapi_improvement(data[3])
        except:
            print("예외가 한번더 발생. 이 데이터는 skip")

    print(data[0])
    print(data[3])
    print(type_response['choices'][0]['message']['content'])
    print(improvement_response['choices'][0]['message']['content'])

    insert_query = "INSERT INTO ttobak_badreview (id, productName, review, gpt_answer, type, improvement) VALUES (%s, %s, %s, %s, %s, %s)"
    insert_data = (data[0], data[1], data[3], data[4], type_response['choices'][0]['message']['content'], improvement_response['choices'][0]['message']['content'])
    cursor.execute(insert_query, insert_data)
    dbConn.commit()


if __name__ == "__main__":
    count_query = "select count(*) from ttobak_review"
    cursor.execute(count_query)
    count = cursor.fetchone()

    for i in range(1, count):
        gpt_badreview(i)
        
    cursor.close()
    mysqlDbClose(dbConn)