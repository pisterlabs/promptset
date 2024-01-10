import sys
sys.path.append(r'C:\ReviewService\package')

from dbconn import mysqlDbConnection, mysqlDbClose
import openai
import pandas as pd

key_file = open('C:\ReviewService\GptApiService\openai_apikey.txt','r', encoding='UTF8')
openai.api_key = key_file.readline()
key_file.close()

dbConn = mysqlDbConnection('root', '0000', '127.0.0.1', 3306, 'reviewdb')
cursor = dbConn.cursor()

def call_gptapi(data):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "All you have to do is look at the product reviews and determine whether they are 'very good', 'good', 'average', 'bad' or 'very bad'." }, # 역할지시문은 한국어보다 영어로 작성해야 더 잘 동작함
            {"role": "user", "content": f"{data} 이 후기 내용은 '매우좋음', '좋음', '보통', '나쁨', '매우나쁨' 중에 무엇인가요? 단어로만 답변해주세요."}
        ],
        timeout = 10    
        )
    return response


def gpt_lactofit(csv_len):
	query = f"SELECT name, content FROM lactofit_review ORDER BY id DESC LIMIT {csv_len}"
	cursor.execute(query)

	datas = cursor.fetchall()
	answerlist = []
	i = 0
	for data in datas:
		try:
			response = call_gptapi(data)
			answerlist.append(response['choices'][0]['message']['content'])
		except:
			print("예외가 한번 발생")
			try:
				response = call_gptapi(data)
				answerlist.append(response['choices'][0]['message']['content'])
			except:
				print("예외가 한번 더 발생. 이 데이터는 skip")
				answerlist.append("예외 발생")
		print(i)
		print(data)
		print(answerlist[i])
		i += 1
			
	dbConn.commit() 
	cursor.close()
	mysqlDbClose(dbConn)

	# dataframe으로 변환 후 csv파일로 저장
	data = {"answer":answerlist}

	df = pd.DataFrame(data)

	df.to_csv("C:\ReviewService\GptApiService\gpt_lactofit_answer.csv", encoding = "utf-8-sig")

if __name__ == "__main__":
	query = "SELECT count(*) FROM lactofit_review"
	cursor.execute(query)

	total_rows_num = cursor.fetchone()[0]
	gpt_lactofit(total_rows_num)