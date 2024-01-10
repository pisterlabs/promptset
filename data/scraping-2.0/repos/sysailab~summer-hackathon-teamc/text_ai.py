import openai
from config import OPENAI_KEY

openai.api_key = OPENAI_KEY

MODEL = "gpt-3.5-turbo"

# first_q = "1. 논문의 제목 \n-English : \n-Korean : \n\n\
#     2.논문의 저자명 \n-English : \n-Korean : \n\n\
#         3.논문의 저자소속 \n -Korean : \n\n\
#             4. 논문의 사사문구"

# first_q = "1. 논문의 제목 \n-Korean : \n\n\
#     2.논문의 저자명 \n-Korean : \n\n\
#         3.논문의 저자소속 \n -Korean : \n\n\
#             4. 논문의 사사문구"

first_q = "1. 논문의 제목 \n-English : \n\n\
    2.논문의 저자명 \n-English :  \n\n\
        3.논문의 저자소속 \n -English : \n\n\
            4. 논문의 사사문구"


class text_ai():
    def __init__() -> None:
        pass

    
    
    def main(r_text):
  
        text = ""
        def input_text(r_text):
            response = openai.ChatCompletion.create(
                    model=MODEL,
                    messages=[
                        {"role": "user", "content": f" {first_q} \n지금 너에게 준 형식들로 뒤에 주는 글의 내용을 정리해줘 사사문구 같은 경우에는 이 논문은 혹은 본 연구는 이라는 단어가 들어간 문단을 말해\n{r_text}"}, 
                    ]
            )
            return response.choices[0].message.content
        
        r_text = r_text[:1400] + r_text[len(r_text)-2600:]
        # print(r_text)
        text += input_text(r_text)
        # print(text)

        return text
        

    
