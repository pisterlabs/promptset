import openai, os, sys, time, re
openai.api_key = "put your OpenAI API"  #이곳에 당신의 OpenAI API를 넣어주세요
completion = openai.Completion()

#A: 문자열 지우기
search = "A:"
none = "None"
temp = sys.stdout # 이 부분이 있으면 close시 에러 발생X

# 파일 추가 폴더 경로
#root = "./Real-time Voice Phishing Detection/원본데이터 셋/VP"
#file_save = "./Real-time Voice Phishing Detection/보이스피싱 가능도/VP"


#root = "./Real-time Voice Phishing Detection/원본데이터 셋/NonVP"
#file_save = "./Real-time Voice Phishing Detection/보이스피싱 가능도/NonVP"


file_save = "./openai-quickstart-python-master/데이터 분할 저장/"
root = "./openai-quickstart-python-master/데이터 분할"
#데이터가 분할된 데이터셋을 데이터 분할 저장에 복사하여 붙여넣기 해주세요


#폴더 안에 있는 파일수 함수
files = []
walk = [root]
file_name = []
file_name = os.listdir(root) #파일 이름만 저장

#파일의 개수 확인 변수
file_num = 0
user_send = []
user_receive = []


#문자를 몇 개씩 나눌지 정하는 변수
split = int(100)

#문장 임시저장 리스트
tmp = []

#전화 step 리스트 크기 지정변수
step_size = int(5)


#보이스 피싱 예제 질문
basic = '''보이스 피싱 대화일 가능성을 0에서 10사이의 정수로 대답해줘. 예문은 답변 형식을 알려주려는 것이고 그 밑줄 아래의 대화를 판단해줘.  

예문 시작
Q: 참가자1: 안녕하세요 
참가자2: 네 안녕하세요 
A: 0. 보이스 피싱 가능성 매우 낮음

Q: 참가자1: 통장과 보안카드를 스캔해서 문자로 보내주세요.
참가자2: 네 알겠습니다.
A: 10. 보이스 피싱 가능성 매우 높음
예문 끝
------------------------------------------------------------------------
Q: '''



last_sentence = str("\n A:")

#open-AI API 통신
def run_openai_chatbot(question, crime_possibility):
    prompt_initial = question
    prompt = basic + prompt_initial + last_sentence
    #print(prompt)
    response = completion.create(
        prompt=prompt, 
        model="text-davinci-003",
        #stop=stop, 
        max_tokens=1,
        temperature=0,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        best_of=1
    )
    #print(response)
    answer = response.choices[0].text.strip()
    #history = prompt + answer

    #print('question: %s\n answer: %s\n' % (question, answer))    
    #print(answer)
    crime_possibility.append(answer)
    return answer, crime_possibility


#폴더 안에 있는 파일수 함수
def read_inside_floder(file_num, files, walk):
    while walk:
        folder = walk.pop(0)+"/"
        items = os.listdir(folder) # items = folders + files
        for i in items:
            file_num = file_num +1
            i=folder+i # 파일 경로와 파일이름 맵핑 리스트
            (walk if os.path.isdir(i) else files).append(i)
    #print(files[4])
    

# 맵핑된 리스트 파일 내용 열기 함수
def read_file(files):
    count = 0
    for i in range(len(files)):
        crime_possibility = [] #범죄 가능성 0~10 리스트 초기화
        #100개씩 문장을 나눠 저장할 리스트
        call_contents = []
        print(i, " ",crime_possibility)
        #전화 step 리스트
        call_step = []
        if os.path.exists(files[i]):
            fr = open(files[i], "r", encoding = 'utf-16')
            #lines = fr.readlines()
            lines = fr.read()
            
            #파일 내 단어의 개수 파악
            num = len(lines)

            #전체 문장 100개씩 나눠서 몇개의 리스트 크기로 나눌지 계산 변수
            data = int(num/split)
            #리스트의 크기를 정했다면 전체 문장이 그룹을 1을 더하여 나머지 남은 문장 for문 
            answer_plus = data + 1
            #전체 문장 100개씩 나눠서 남은 문장의 개수 변수
            remainder = int(num%split)
            #blank_remove = [value for value in lines if value != "\n"]
            
            words = 0 #단어 100개씩 나눠서 리스트에 넣을 위치 변수
            for main in range(answer_plus):
                #변수 초기화
                tmp = []
                blank_remove = []
                #처음 전체 문장 100개씩을 리스트에 저장 부분
                if main < data:
                    for l in range(split):
                        #임시로 tmp 리스트에 100개의 문자를 저장
                        tmp.append(lines[words])
                        #print(words, ":몇 번째 줄 ", main, "X", words, "=")
                        #print(words, ":몇 번째 줄 ", lines[words])
                        words += 1
                
                #남은 문장을 리스트에 저장 부분
                elif main >= data:
                    for l in range(remainder):
                        tmp.append(lines[words])
                        #call_contents.append(lines[words])
                        #print(words, ":몇 번째 줄 ", main, "X", words, "======")
                        #print(words, ":몇 번째 줄 ", lines[words])
                        words += 1

                #blank_remove 리스트 변수에 100개의 문자를 합치기
                blank_remove = "".join(tmp)
                #call_contents 리스트에 저장
                call_contents.append(blank_remove)

            #print(call_contents[0])
            

            #변수 초기화
            tmp = []
            blank_remove = []
            # 윈도우 슬라이딩
            out = int(1)
            for j in range(len(call_contents)):
                if j < step_size:
                        tmp = call_contents[0:j+1]
                        blank_remove = "".join(tmp)
                        call_step.append(blank_remove)

                elif j >= step_size:
                    tmp = call_contents[out:j+1]
                    blank_remove = "".join(tmp)
                    call_step.append(blank_remove)
                    out += 1
            #question = call_step[7]
            #print(question)
 
            #print(num)
            #print(answer)
            #print(remainder)

                 
            #step별로 검색
            for j in range(len(call_step)):
                question = call_step[j]
                #print("*************************************")
                #print("Time of Q/A: ",i+1)
                answer, crime_possibility = run_openai_chatbot(question=question, crime_possibility=crime_possibility)
                print("{}:{}".format(j,crime_possibility))
                #print(answer)

            #for i in range(len(lines)):
            #    print(lines[i], end='')
            #print(answer)
            #print(remainder)
            #print(call_contents[0])

            fr.close()

            #범죄 가능성 리스트 안에 A: 문자열 제거
            for j, word in enumerate(crime_possibility):
                if search in word:
                    crime_possibility[j] = word.strip(search)
            print("{}:{}".format(i,crime_possibility))
            #범죄 가능성 0~10 API 저장
            save_name = file_name[count]
            # 보이스피싱 가능도를 txt파일로 저장하는 부분
            f = open(file_save + save_name, 'w', encoding='utf-8')
            my_string = ', '.join(str(j) for j in crime_possibility)
            my_string = '[' + my_string + ']'
            f.write(my_string)
            f.close()
            count += 1
            del crime_possibility
            time.sleep(5)

        else:
            print("파일이 존재하지 않네요.")
            


#폴더 안에 있는 파일수 함수 호출
read_inside_floder(file_num, files, walk)
#파일내용 읽고 수신인, 발신인 별로 리스트 호출
read_file(files)

