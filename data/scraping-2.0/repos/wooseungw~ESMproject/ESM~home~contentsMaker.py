

import sys
import os
import openai
import json
from django.core.cache import cache




#openai.organization = "org-RW97zLho4qp0kezjTGL3HLRb"
mykey = "-"
openai.api_key = f"{mykey}"


class contentsMaker:
    
    
    index_list = []
    content_list = []
    title = "."
    fname = "."
    ftype = "."
    keywords = "."
    
    def response(message_list):
    
        answer = openai.ChatCompletion.create(model="gpt-3.5-turbo-0301", messages=message_list)
        return answer['choices'][0]['message']['content']


    #텍스트 파일을 리스트에 저장 
    def txttolist():
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        txt_folder_path  = os.path.join(f'{BASE_DIR}', 'txt')
        
        txt_files = os.listdir(txt_folder_path)
        for txt_file in txt_files:
            txt_path  = os.path.join(txt_folder_path, txt_file)
            with open(txt_path, 'r',encoding='utf-8') as file:
                text = file.read()
                input_list = text.split('\n\n\n')
            input_list.append("다음입력에 새로운 파일입력됨")
        
        contentsMaker.fname,contentsMaker.ftype = txt_file.split(".")
        
        
        return input_list
    
    def index():
        message_list = []    
        input_list = contentsMaker.txttolist()   
        
        #index 생성 프롬프트 추가
        message_list.append(
            {"role": "system",
            "content": f"어떤 글들이 입력될꺼야 이 글들을 잘 읽어보고 '목차를 출력해줘'라고 하면 목차를 출력해줘. 관련된 지식을 쉽게 습득할수 있도록 도와주는 목차를 작성해줘.서론,입력된 주제에 관한 이해를 돕는글, 주제를 쉽게 알 수 있는 키워드,배경지식,결론 -예시- 주제: 자바작동원리 목차가 될수 있는것들: 자바가 어떻게 작동하는지와 관련된 개념인 힙 메모리, 참조에 의한 호출, 모듈, 클래스,등등과 실제작동을 알 수 있는 예제, 작동원리의 장점, 작동원리의 단점, 작동 방식의 최상의 시나리오, 작동방식의 최악의 시나리오, 빈번하게 일어아는 시나리오, 등등.출력 형식: [서론]\n\n[Chapter.1]\n-주제에 관한 이해\n-주제에 관한 이해2\n주제에 관한 이해3\n\n[Chapter.2]\n-소목차1\n-소목차2\n\n[Chapter.3]\n-소목차1\n-소목차2\n\n-결론.\n\n———— 다음 입력에 주제가 입력될꺼야. 알겠다면 '주제 및 내용을 입력해주세요!'라고 출력해줘."})
        
        #메세지 리스트가 비어있을때 까지 출력 일단 하나만 받기
        i=0           
        for intxt in input_list:
            if i ==1:
               break 
            message_list.append({"role": "system","content": f"{intxt}"})
            response =  contentsMaker.response(message_list)   
            i +=1     
            
        message_list.append({"role": "user", "content": f" 이 문서에서의 키워드 5~10개를 콤마로 구분해서 출력해줘"})
        keywords = contentsMaker.response(message_list)
        message_list.append({"role": "assistant", "content": f"{keywords}"})
        
        
            
        message_list.append({"role": "user", "content": f"목차 출력형식을 활용해서 이 문서의 지식을 이해하는데 도움이 될 목차를 출력해줘. 키워드를 활용하면 좋을꺼같아. 만약 다른 챕터의 목차와 겹치는 제목이 있다면 이번챕터와 관련된 목차명으로 바꾸던가 아니면 그냥 지워줘."})
        response =  contentsMaker.response(message_list)        
        message_list.append({"role": "assistant", "content": f"{response}"})
        
        message_list.append({"role": "user", "content": f" 적절한 제목을 출력해줘"})
        title = contentsMaker.response(message_list)
        
        
        contentsMaker.keywords = keywords
        contentsMaker.title = title
        #message_list.append({"role": "assistant", "content":f"{indexs}"})    
        
        #print(response)
        
        return response

    def contents(indexs):
        
        contents_list = []
        #콘텐츠 생성 프롬프트
        message_list = [
            {"role": "system",
            "content":f"목차정보가 입력될꺼야. 목차정보를 천천히 분석하고, 블로그 글을 작성해줘.목차별 내용을 적어도 3줄이상의 문장이어야해. 설명할때 예제를 같이 사용하면 더 좋을거같아. 입력 예)2. 머신러닝과 딥러닝 기술의 원리와 활용\n- 머신러닝 기초 개념과 알고리즘\n- 신경망과 딥러닝의 원리와 활용\n- 딥러닝을 활용한 이미지 인식, 자연어 처리 등 다양한 분야 적용 사례-----출력 예)[Chapter.1]머신러닝과 딥러닝 기술의 원리와 활용\n\n-머신러닝 기초 개념과 알고리즘\n(머신러닝은 인공지능의 한 분야로, 컴퓨터가 데이터를 통해 학습하고 예측, 분류, 군집화 등의 작업을 수행할 수 있도록 하는 알고리즘과 기술들을 연구하는 분야입니다. 머신러닝의 목표는 명시적인 프로그래밍 없이 컴퓨터가 데이터로부터 패턴을 학습하고 일반화하여 새로운 데이터에 대한 의사결정을 내릴 수 있게 하는 것입니다.\n머신러닝에는 여러 가지 기초 개념과 알고리즘이 있습니다:\n지도 학습(Supervised Learning): 입력 데이터와 그에 상응하는 정답 레이블이 주어지고, 알고리즘이 이를 학습하여 새로운 데이터에 대한 예측을 수행합니다. 지도 학습의 대표적인 알고리즘으로는 선형 회귀(Linear Regression), 로지스틱 회귀(Logistic Regression), 서포트 벡터 머신(SVM), 결정 트리(Decision Tree) 등이 있습니다.\n비지도 학습(Unsupervised Learning): 입력 데이터만 주어지고 정답 레이블이 없는 상태에서 패턴이나 구조를 찾는 학습 방법입니다. 비지도 학습의 대표적인 알고리즘으로는 클러스터링(Clustering), 주성분 분석(PCA), 자동 인코더(Autoencoder) 등이 있습니다.\n강화 학습(Reinforcement Learning): 에이전트가 환경과 상호 작용하며 보상을 최대화하는 행동을 학습하는 방법입니다. 강화 학습은 마르코프 결정 과정(MDP)과 Q-러닝(Q-Learning), 딥 Q-네트워크(Deep Q-Network, DQN) 등의 개념과 알고리즘을 사용합니다.\n딥러닝(Deep Learning): 머신러닝의 한 분야로, 인공 신경망(Artificial Neural Networks, ANN)을 기반으로 복잡한 패턴을 학습할 수 있는 알고리즘입니다. CNN(Convolutional Neural Networks), RNN(Recurrent Neural Networks), LSTM(Long Short-Term Memory), GAN(Generative Adversarial Networks) 등 다양한 구조와 알고리즘이 존재합니다.)\n\n-신경망과 딥러닝의 원리와 활용\n(내용)\n\n- 딥러닝을 활용한 이미지 인식, 자연어 처리 등 다양한 분야 적용 사례\n(내용)----- 글쓰기 방식:한 문단은 '\n\n'로 구분, 각 챕터별로 짜임새있는 글쓰기, 글작성이 마무리 된 후 퇴고작업으로 글 완성도 높히기 내외 사용.———— 반드시 이 형태와 요청을 기억해줘."}    ]
        #주제 입력, 프롬프트 입력, 인덱스 받기,목차수정

        
            
        #for index in indexs:
            
        message_list.append({"role": "user", "content": f"{indexs}이 목차정보를 잘 설명할수 있는 글을 작성해줘 출력 예시를 참고해줘. 이미 서론을 적었다면 더이상 적지 않아도 돼. 적어도2500토큰 이상 사용해줘.4090토큰보다 적게 사용해야해. 모든 목차가 골고루 설명될수 있으면 좋겠어 서론과 결론은 조금 짧아도 괜찮아."})
            #목차마다 정보생성
        contents =  contentsMaker.response(message_list)
            
            #message_list.append({"role": "assistant","content":"관련된 정보 출력."})
            #컨텐츠 저장
            
        contents_list = contents.split('\n\n')
        
        return contents_list
        

            


# index = contentsMaker.index()
# title = contentsMaker.title
# print(title)
# print("-------------------------------------------------------------")
# print(index)
# print("-------------------------------------------------------------")
# contents = contentsMaker.contents(index)
# print(contents)
