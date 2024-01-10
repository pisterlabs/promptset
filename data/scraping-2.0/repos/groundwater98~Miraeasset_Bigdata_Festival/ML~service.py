from Summary import chatGPT
from Recommendation import recommend
from Prediction import predict
from Seq2seq import Seq2seqLSTM
from typing import Tuple
import openai
import pdb


def get_menu(question) -> Tuple[str, str, bool]:
    # 사용자에게서 어떤 서비스를 원하는지 가져온다.
    # chatGPT를 이용해서 대화형에 가깝게 만든다.
    My_OpenAI_key = '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'
    openai.api_key = My_OpenAI_key
    humor = False
    """
    # 카카오 챗봇과 연결 안했을 때는 cli로 질문 입력받음.

    print(f"Hi I'm Sacretary for your Investment.\n What can I help you?")
    print(f"=========Here is services we provide=========")
    print(f"1. Information Summary\n2. Prediction\n3. Recommendation\n4. Quit")
    print(f"=============================================\n")
    print(f"You can also give the sentence.")
    question = input(f"Please write the service name, number you want or the sentence. ")
    """
    question = question

    role = """Your role is to know well what kind of service the other person wants from what they say. 
    The service must be one of Information Summary, Stock Price Prediction, or Stock Recommendation.
    If the service is not one of these, answer in the second format. If yes, answer in the first format.

    first form
    service:

    second form
    I'm not sure about that part."""

    # 서비스가 불명확하면, 그냥 유머스럽게 대답.
    print(f"Analyzing what you want ...\n")
    # messages는 대화의 흐름을 기억할 수 있게 해준다.
    # system, user, assistant, user, assistant ... 반복해나가면서 대화 흐름 기억 가능.
    # role은 총 3가지가 있다. system, user, assistant <-- 이건 스펠링이 확실히 맞는지 모름 
    # system은 chatGPT의 역할 부여, user는 우리의 질문, assistant는 chatGPT의 답변
    answer = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
                {"role": "system", "content": role},
                {"role": "user", "content": question},
            ]
    )
    answer = answer.choices[0]['message']['content']
    if answer.find("part") == -1:
        # 제대로 된 서비스 입력한 경우
        answer = answer.split(":")[1].strip() # 고객이 원하는 서비스
    else:
        # 제대로 된 입력 안 한 경우
        role = """You are very humorous and good at telling jokes. Answer questions you find difficult to answer with a joke."""

        answer = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                    {"role": "system", "content": role},
                    {"role": "user", "content": question},
                ]
        )
        answer = answer.choices[0]['message']['content']
        humor = True
    
    #pdb.set_trace()
    if not humor:
        print(f"You want the service, \"{answer}\" Right? I will give you the answer. Please wait for seconds...\n")
    else:
        print(answer)
    return answer, question, humor


def kakao(question):
    service, question, humor = get_menu(question)
    if not humor:
        if service == 'Information Summary':
            print(f"Labeling & Information Summary ...")
            role = """Your role is to find out which company the other person wants 
            news from and what period of time they want news from the sentences they are speaking. 
            The answer format should be as follows.

            form:
            Company you want to know:
            Period you want to know:"""

            answer = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                        {"role": "system", "content": role},
                        {"role": "user", "content": question},
                    ]
            )

            answer = answer.choices[0]['message']['content']
            #pdb.set_trace()
            answer = answer.split("\n")
            company = answer[0].split(":")[1].strip()
            period = answer[1].split(":")[1].strip()
            question = f"{company}의 {period}전 뉴스를 알고싶어."

            # 실제로 sql로 db접근은 미구현 했기때문에
            # news는 None 객체
            news = Seq2seqLSTM.get_sql(question)
            answer = chatGPT.labeling_module(news) 
        elif service == 'Stock Price Prediction':
            role = """You will receive a statement asking for a stock price prediction, 
            and your role is specialized in finding out which stock you want to predict. 
            Please answer in the first format below. If the forecast period and stock are not specified, 
            please answer in the second format.

            First answer format:
            Predicted stocks:
            Forecast Period: only days

            Second response format:
            There is insufficient information to satisfy your needs. Please tell me specifically what stock price you want and when.
            Example) Predict Apple's stock price tomorrow."""

            answer = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                        {"role": "system", "content": role},
                        {"role": "user", "content": question},
                    ]
            )
            answer = answer.choices[0]['message']['content']
            if answer.find("First answer format") == -1:
                # 첫번째 대답 형식 아닌 경우: 예측 종목도 없고, 기간도 없는 경우
                #pdb.set_trace()
                answer = answer.split("\n")
                stock = answer[0].split(":")[1].strip()
                period = int(answer[1].split(":")[1].split()[0])
                answer = f"I want to know the price of {stock} stock for {period} days."
            else:
                # 두번째 대답 형식 아닌 경우: 예측, 기간 특정된 경우
                pass
            answer = predict.predict(stock, period)
        elif service == 'Stock Recommendation':
            user_inform = "우리 서비스 이용중인 고객의 정보를 불러와야 함."
            answer = recommend.recommend(user_inform)
        else:
            print(f"\n\nPlease write the correct service!!")
        refined_answer = chatGPT.Refine_module(answer)
        kor2eng = chatGPT.Kor2Eng(refined_answer)
        print("="*30)
        print(f"Here's our final answer:")
        print(kor2eng)
        return kor2eng
    else:
        return service
        

if __name__ == '__main__':
    kakao()
    print(f"\n\nThank you for your using.")