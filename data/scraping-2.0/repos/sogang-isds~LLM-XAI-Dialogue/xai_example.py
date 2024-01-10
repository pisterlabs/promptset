import logging

import gradio as gr
from langchain.chat_models import ChatOpenAI

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

# api_key = os.getenv("OPENAI_API_KEY")
key = 'OPENAI_API_KEY'
api_key = 'sk-' + key + 'bkFJm28ZY54dRNHk3u5edkod'

chat_llm = ChatOpenAI(model_name="gpt-3.5-turbo", api_key=api_key)


class ChatGPT:
    def __init__(self):
        self.messages = []

    def request(self, prompt):
        self.messages.append(HumanMessage(content=prompt))
        ai_message = chat_llm(self.messages)
        self.messages.append(ai_message)
        response_message = ai_message.content

        return response_message

    def reset(self):
        logging.info(f'before reset: {len(self.messages)} messages')
        self.messages = []
        logging.info(f'after reset: {len(self.messages)} messages')


class XAILawPrompt:
    def __init__(self, article=None, commercial_law=None):
        if article:
            self.article = article
        if commercial_law:
            self.commercial_law = commercial_law

    def set_article(self, article):
        self.article = article

    def set_commecial_law(self, commercial_law):
        self.commercial_law = commercial_law

    def generate_question_prompt(self, question):
        question_prompt = f"""다음 정관 문서를 읽고 질문에 답하세요. 답변은 정관에 기반해서 간결하게 작성해 주세요. 정관에 없는 내용일 경우 없다고 알려주세요.
{self.article}

질문: {question}
답변: 내용(조항)
"""
        return question_prompt

    def generate_analytic_prompt(self, question, result, paragraph_key):
        question_idx = related_questions[paragraph_key].index(question)
        commercial_law = related_commercial_law[paragraph_key][question_idx]

        prompt = f"""정관의 내용은 아래와 같습니다.
{self.article}

아래는 정관에 대한 질문과 답변입니다.

질문: {question}
답변: {result}

이 질문과 관련된 법령은 아래와 같습니다.
상법: {commercial_law}

변호사라고 생각하고 상법에 기반하여 정관의 규정이 적법한지 조언을 상세하게 작성해 주세요.

변호사 조언:
"""

        return prompt


related_commercial_law = {
    "제4장 주주총회": [
        '제365조(총회의 소집) ①정기총회는 매년 1회 일정한 시기에 이를 소집하여야 한다. ②연 2회 이상의 결산기를 정한 회사는 매기에 총회를 소집하여야 한다. ③임시총회는 필요있는 경우에 수시 이를 소집한다.',
        '제365조(총회의 소집) ③임시총회는 필요있는 경우에 수시 이를 소집한다.',
        '제365조(총회의 소집) ①정기총회는 매년 1회 일정한 시기에 이를 소집하여야 한다. ②연 2회 이상의 결산기를 정한 회사는 매기에 총회를 소집하여야 한다. ③임시총회는 필요있는 경우에 수시 이를 소집한다. 제362조(소집의 결정) 총회의 소집은 본법에 다른 규정이 있는 경우 외에는 이사회가 이를 결정한다.',
        '제366조의2(총회의 질서유지) ①총회의 의장은 정관에서 정함이 없는 때에는 총회에서 선임한다. ②총회의 의장은 총회의 질서를 유지하고 의사를 정리한다. ③총회의 의장은 고의로 의사진행을 방해하기 위한 발언ㆍ행동을 하는 등 현저히 질서를 문란하게 하는 자에 대하여 그 발언의 정지 또는 퇴장을 명할 수 있다.',
        '제366조의2(총회의 질서유지) ①총회의 의장은 정관에서 정함이 없는 때에는 총회에서 선임한다. ②총회의 의장은 총회의 질서를 유지하고 의사를 정리한다. ③총회의 의장은 고의로 의사진행을 방해하기 위한 발언ㆍ행동을 하는 등 현저히 질서를 문란하게 하는 자에 대하여 그 발언의 정지 또는 퇴장을 명할 수 있다.'
    ],
    "제5장 이사·이사회": [
        '제382조(이사의 선임, 회사와의 관계 및 사외이사) ① 이사는 주주총회에서 선임한다.',
        '제368조(총회의 결의방법과 의결권의 행사) ①총회의 결의는 이 법 또는 정관에 다른 정함이 있는 경우를 제외하고는 출석한 주주의 의결권의 과반수와 발행주식총수의 4분의 1 이상의 수로써 하여야 한다. <개정 1995. 12. 29.>',
        '제382조의2(집중투표) ①2인 이상의 이사의 선임을 목적으로 하는 총회의 소집이 있는 때에는 의결권없는 주식을 제외한 발행주식총수의 100분의 3 이상에 해당하는 주식을 가진 주주는 정관에서 달리 정하는 경우를 제외하고는 회사에 대하여 집중투표의 방법으로 이사를 선임할 것을 청구할 수 있다.'
    ]
}

related_questions = {
    "제4장 주주총회": [
        '정기주주총회를 개최하는가?',
        '임시주주총회를 개최하는가?',
        '정기주주총회는 언제 소집하나?',
        '주주총회 의장은 누구인가?',
        '주주총회 의장이 유고 또는 부재시에 어떻게 하는가?'
    ],
    "제5장 이사·이사회": [
        '이사는 어디에서 선임하는가?',
        '이사를 선임하기 위한 정족수는 얼마인가?',
        '이사 선임에 집중투표제를 배제하였는가?'
    ]
}
example_paragraph = {
    "제4장 주주총회": """제 4 장 주주총회 
    
제17조 (소집시기) 
① 회사의 주주총회는 정기주주총회와 임시주주총회로 한다. 
② 정기주주총회는 매사업연도 종료 후 3월 이내에, 임시주주총회는 필요에 따라 소집한다. 

제18조 (소집권자) 
① 주주총회의 소집은 법령에 다른 규정이 있는 경우를 제외하고는 이사회 또는 이사회로부터 위임 받은 위원회의 결의에 따라 대표이사가 소집한다. 
② 대표이사 유고시에는 이사회에서 별도로 정한 순서에 따라 그 직무를 대행한다. 

제19조 (소집통지 및 공고) 
① 주주총회를 소집함에는 그 일시, 장소 및 회의의 목적사항을 총회일 2주간 전에 각 주주에게 서면으로 통지를 발송하거나 각 주주의 동의를 받아 전자문서로 통지를 발송하여야 한다. 
② 의결권 있는 발행주식총수의 100분의 1 이하의 주식을 소유한 주주에 대한 소집통지는 2주간 전에 주주총회를 소집한다는 뜻과 회의 목적사항을 서울특별시에서 발행하는「매일경제신문」과 「한국경제신문」에 2회 이상 공고하거나 금융감독원 또는 한국거래소가 운영하는 전자공시시스템에 공고함으로써 제1항의 소집통지에 갈음할 수 있다. 

제20조 (소집지) 
주주총회는 본점소재지에서 개최하되 필요에 따라 이의 인접지역에서도 개최할 수 있다. 

제21조 (총회 의장) 
① 주주총회의 의장은 이사회에서 정한 이사로 한다 
② 이사회에서 정한 이사의 유고시에는 이사회에서 별도로 정한 순서에 따라 그 직무를 대행한다. 

제22조 (의장의 질서유지권) 
① 주주총회의 의장은 고의로 의사진행을 방해하기 위한 발언·행동을 하는 등 현저히 질서를 문란하게 하는 자에 대하여 그 발언의 정지 또는 퇴장을 명할 수 있다. 
② 주주총회의 의장은 의사진행의 원활을 기하기 위하여 필요하다고 인정할 때에는 주주의 발언의 시간 및 회수를 제한할 수 있다. 
    """,
    "제5장 이사·이사회": """제 5 장 이사·이사회 

제29조(이사의 수) 
① 본 회사의 이사는 3인 이상 9인 이하로 한다. 
② 이사는 사내이사, 사외이사와 그 밖에 상시적인 업무에 종사하지 아니하는 이사로 구분하고, 사외이사는 3인 이상으로 하되, 이사 총수의 과반수로 한다. <개정 1997.5.31, 1998.5.30, 2000.5.27, 2004.6.4, 2012.6.5, 2013.6.27, 2017.3.24> 

제30조(이사의 선임) 
① 이사는 주주총회에서 선임한다. <개정 2000.5.27> 
② 2인 이상의 이사를 선임하는 경우에도 상법 제382조의 2에서 규정하는 집중투표제를 적용하지 아니한다. [신설 1999.5.29］ 

제30조의2(임원 후보의 추천) 
① 임원후보추천위원회는 금융회사의 지배구조에 관한 법률, 상법 등 관련 법규에서 정한 자격을 갖춘 자 중에서 임원(대표이사, 사외이사, 감사위원에 한함) 후보를 추천한다. [신설 2012.6.5, 개정2017.3.24］ 
② 임원 후보의 추천 및 자격심사에 관한 세부적인 사항은 임원후보추천위원회에서 정한다. [신설 2012.6.5, 개정2017.3.24] 

제31조(이사의 임기) 
① 이사의 임기는 2년 이내로 하되, 연임할 수 있다. 다만, 사외이사는 회사에서 사외이사로 6년 이상 재직할 수 없고, 회사의 계열회사에서 사외이사로 재직한 기간을 합하여 9년 이상 재직할 수 없다. <개정 2005.3.10, 2010.5.28, 2012.6.5, 2015.3.27, 개정2017.3.24> 
② 제1항의 임기가 최종 결산기 종료 후 해당 결산기에 관한 정기주주총회 전에 만료될 경우에는 그 정기주주총회의 종결시까지 그 임기가 연장된다. <개정 1996.5.25, 2000.5.27, 2005.5.27, 2013.6.27, 2014.12.17> 

제32조(이사의 보선) 
① 이사 중 결원이 생긴 때에는 주주총회에서 이를 선임한다. 그러나 이 정관 제29조에서 정하는 원수를 결하지 아니하고 업무수행상 지장이 없는 경우에는 그러하지 아니한다. <개정 2000.5.27, 2012.6.5> 
② 보궐선임된 이사의 임기는 제31조에 따른다. <개정 2000.5.27>"""
}


def main():
    prompt = None

    with gr.Blocks() as demo:
        prompt_manager = XAILawPrompt()

        def click_button(msg, paragraph_key):
            print(f"msg: {msg}, paragraph_key: {paragraph_key}")
            sentences = msg.split('\n')
            print(sentences)

            return gr.Radio(choices=related_questions[paragraph_key])

        def click_button2(question, result, paragraph_key):
            print(f"question: {question}, result: {result}")

            prompt = prompt_manager.generate_analytic_prompt(question, result, paragraph_key)
            print(prompt)

            chat_gpt = ChatGPT()
            result = chat_gpt.request(prompt)

            return gr.Label(value=result)

        def click_radio(question, text):
            print('click_radio:', question)

            prompt = prompt_manager.generate_question_prompt(question)
            print(prompt)

            chat_gpt = ChatGPT()
            result = chat_gpt.request(prompt)

            return gr.Label(value=result)

        def select_radio():
            print('select_radio')
            return gr.Label(value=None)

        def select_dropdown(choice):
            print(f'select_dropdown: {choice}')
            text = example_paragraph[choice]
            prompt_manager.set_article(text)
            return text

        gr.Markdown("""# 생성형 AI 기반 정관 검토 서비스""")
        gr.Markdown("""## 정관 문단 선택""")
        gr.Markdown("""분석을 하고자 하는 정관 문단을 선택하세요.""")

        dropdown = gr.Dropdown(choices=['제4장 주주총회', '제5장 이사·이사회'], label='정관 문단')

        gr.Markdown("""## 정관 검토""")
        text1 = gr.Text(prompt, label='정관 내용', max_lines=10)
        btn1 = gr.Button("정관 체크리스트 분석")

        dropdown.select(select_dropdown, inputs=dropdown, outputs=text1)

        gr.Markdown("""## 체크리스트 분석""")
        gr.Markdown("""위 정관 내용에서 분석되어야 할 항목을 출력합니다.""")

        question_radio = gr.Radio(label='변호사 체크리스트', info='아래 버튼 중 하나를 선택해 주세요.', choices=[], interactive=True)

        output_label = gr.Label(label='분석된 답변')
        btn2 = gr.Button("Analytic Statement 분석")

        output_label2 = gr.Label(label='Analytic Statement 분석 결과')

        question_radio.input(fn=click_radio, inputs=[question_radio, text1], outputs=[output_label], queue=False)
        question_radio.select(fn=select_radio, outputs=output_label2, queue=False)

        btn2.click(fn=click_button2, inputs=[question_radio, output_label, dropdown], outputs=output_label2,
                   queue=False)

        btn1.click(fn=click_button, inputs=[text1, dropdown], outputs=question_radio, queue=False)

    demo.launch()


if __name__ == '__main__':
    main()
    # exit()
