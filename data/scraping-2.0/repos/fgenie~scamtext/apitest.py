import openai
import time

# def timeit(func):
#     def wrapper(*args, **kwargs):
#         start_time = time.time()
#         result = func(*args, **kwargs)
#         end_time = time.time()
#         print(f"Function {func.__name__} took {end_time - start_time:.6f} seconds to execute.")
#         return result
#     return wrapper



def main():
    openai.api_key = open('apikey.txt').readlines()[0].strip()
    
    inp = '''normal text:
[Web발신]
자동매매 
3만원으로 체험중
평균 5,710,000  마감중
https://ko.gl/VPLwd


spam text:
[Web발신]
(광고) 글라스바바안경 안산중앙점

웰컴할인쿠폰 2만원권 당첨!!

▶ 쿠폰번호 : 837282934983
▶ 쿠폰 유효기간 : 2023-02-01~03-31

신년특별이벤트로 글라스바바 고객님들 중 추첨에 당첨되신 분들에게 할인쿠폰 2만원권을 보내드립니다.

<웰컴할인쿠폰 2만원권 >
※ 안경/선글라스 구매 시 첨부된 쿠폰을 제시해주시면 2만원 할인적용됩니다.
※ 본인에 한해 1회 사용가능
※ 유효기간이 지난 쿠폰은 사용불가

최상의 비전케어와 기분좋은 혜택으로 고객감동을 더해 가겠습니다.

글라스바바안경 안산중앙점 031-485-5903

무료거부 080-855-3567

Write 10 various python function called "detect_spam".
The function need to detect spam message by its pattern elucidated in the examples above while avoiding to filter normal message.
The functions need to return True when the message is considered a spam (Otherwise, return False). 

'''
    
    response1 = openai.ChatCompletion.create(
                        model = 'gpt-4',
                        messages = [
                            {'role': 'system', 'content': 'you are a helpful assistant.'},
                            {'role': 'user', 'content': inp},
                        ]
    )
    response2 = openai.ChatCompletion.create(
                        model = 'gpt-3.5-turbo',
                        messages = [
                            {'role': 'system', 'content': 'you are a helpful assistant.'},
                            {'role': 'user', 'content': inp},
                        ]
    )
    print(response1['choices'][0]['message']['content'])
    print(response2['choices'][0]['message']['content'])
    print("gpt4, gpt3.5")




if __name__ == "__main__":
    main()