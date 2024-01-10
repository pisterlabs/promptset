import os
from datetime import datetime, timedelta

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate
)
from langchain.chains import create_extraction_chain

api_key = os.environ.get("OPENAI_API_KEY")

nlu_prompt_text = """당신은 일정관리 시스템 입니다. 일정관리를 위해 필요한 slot, value를 추출합니다.

event_name이 확실하지 않을 땐 추출하지 않습니다.
time은 HH:MM 형식으로 출력합니다.
date는 YYYY-MM-DD 형식으로 출력합니다.

현재날짜: {today}
"""

dst_prompt_text = """당신은 일정관리 시스템의 Dialog State Tracker 입니다. 
일정관리를 위해 nlu_result를 분석하여 dialog_state를 업데이트 하세요.
read를 수행하기 위해서는 1개 이상의 db가 필요합니다.
update, delete를 수행하기 위해서는 오직 1개의 db만 필요합니다.
업데이트 된 dialog_state에는 현재 dialog_state db에 있는 값 중 slot, value 조건에 맞는 값만 남겨놓습니다.


# data
nlu_result: {nlu_result}
dialog_state: {dialog_state}
현재날짜: {today}

# 응답
- 업데이트된 dialog_state 를 dict 형태로 출력
"""

dp_prompt_text = """당신은 일정관리 시스템의 Dialog Policy 입니다.
dialog_state를 분석하여 system_action을 결정하세요.

# data
dialog_state: {dialog_state}
현재날짜: {today}

# 응답
- 형식: system_action을 dict 형태로 출력
- keys:
  - system_action: (Required) 
    - inform: 정보를 알려줄 때
    - request: 부족한 정보를 물어볼 때 slot을 함께 출력
    - run_action: 필요한 정보를 충족할 경우 dialog_state의 action을 수행
  - slot: event_name, date, time (Optional)
  - value: str (Optional)
"""

nlg_prompt_text = """당신은 일정관리 시스템의 Natural Language Generator 입니다.
system_stater값을 이용하여 user에게 자연어 형태로 응답하세요.

# data
system_state: {system_state}
현재날짜: {today}

# 응답
- 자연어 형태로 출력
"""


class PromptAgent:
    def __init__(self, llm, verbose=False):
        self.dp_chain = None
        self.nlu_chain = None
        self.dst_chain = None
        self.nlg_chain = None
        self.llm = llm
        self.verbose = verbose

        self.init_nlu_chain()
        self.init_dst_chain()
        self.init_nlg_chain()
        self.init_dp_chain()

        self.today = datetime.today().strftime("%Y-%m-%d %H:%M")

    def init_nlu_chain(self):
        schema = {
            "properties": {
                "event_name": {"type": "string"},
                "action": {"type": "string", "enum": ["create", "read", "update", "delete", "inform", "request"]},
                "date": {"type": "string", "description": "날짜"},
                "time": {"type": "string", "description": "시간"},
            },
            "required": ["action"],
        }

        nlu_prompt = ChatPromptTemplate(
            messages=[
                AIMessagePromptTemplate.from_template(nlu_prompt_text),
                HumanMessagePromptTemplate.from_template("{user_input}"),
            ]
        )

        self.nlu_chain = create_extraction_chain(schema, self.llm, prompt=nlu_prompt, verbose=True)

    def run_nlu_chain(self, inp):
        response = self.nlu_chain.run({'user_input': inp, 'today': self.today})
        return response[0]

    def init_dst_chain(self):
        dst_prompt = ChatPromptTemplate(
            messages=[
                AIMessagePromptTemplate.from_template(dst_prompt_text),
                # HumanMessagePromptTemplate.from_template("{user_input}"),
            ],
            input_variables=["nlu_result", "dialog_state", "today"],
        )
        self.dst_chain = LLMChain(llm=self.llm, prompt=dst_prompt, verbose=True)

    def run_dst_chain(self, dialog_state, nlu_result):
        response = self.dst_chain.run({'dialog_state': dialog_state, 'nlu_result': nlu_result, 'today': self.today})
        return eval(response)

    def init_dp_chain(self):
        dp_prompt = ChatPromptTemplate(
            messages=[
                AIMessagePromptTemplate.from_template(dp_prompt_text),
                # HumanMessagePromptTemplate.from_template("{user_input}"),
            ]
        )
        self.dp_chain = LLMChain(llm=self.llm, prompt=dp_prompt, verbose=True)

    def run_dp_chain(self, dialog_state):
        response = self.dp_chain.run({'dialog_state': dialog_state, 'today': self.today})
        return eval(response)

    def init_nlg_chain(self):
        nlg_prompt = ChatPromptTemplate(
            messages=[
                AIMessagePromptTemplate.from_template(nlg_prompt_text),
                # HumanMessagePromptTemplate.from_template("{user_input}"),
            ]
        )
        self.nlg_chain = LLMChain(llm=self.llm, prompt=nlg_prompt, verbose=True)

    def run_nlg_chain(self, system_state):
        response = self.nlg_chain.run({'system_state': system_state, 'today': self.today})
        return response

    def update_dialog_state(self, dialog_state, db=None):
        action = dialog_state['action']

        if db:
            dialog_state.update({'db': db})

        if action == 'read':
            pass

        elif action == 'update':
            db = dialog_state['db']

            if len(db) == 1:
                for key, val in db[0].items():
                    if key in dialog_state.keys():
                        if dialog_state[key] == '':
                            dialog_state[key] = val

        return dialog_state

    def update_system_state(self, system_state, dialog_state):
        action = dialog_state['action']
        db = dialog_state['db']
        system_action = system_state['system_action']

        if system_action == 'inform':
            pass
        elif system_action == 'request':
            pass
        elif system_action == 'run_action':
            if action == 'read':
                system_state['db'] = db
            elif action == 'update':
                # update DB
                for key, val in db[0].items():
                    system_state[key] = val
                for key, val in dialog_state.items():
                    if key != 'db':
                        system_state[key] = val

        system_state['action'] = action
        return system_state


def main():
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", api_key=api_key)
    prompt_agent = PromptAgent(llm=llm, verbose=True)

    dialog_state = {'event_name': '', 'action': '', 'date': '', 'time': '', 'db': []}

    date_today = datetime.today().strftime("%Y-%m-%d")
    date_tomorrow = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    schedule_list = [
        {
            'event_name': '산책가기',
            'date': date_today,
            'time': '10:00'
        },
        {
            'event_name': '데이트',
            'date': date_tomorrow,
            'time': '12:00'
        }
    ]

    # Input
    input_list = ['내일 일정을 조회해줘', '일정을 이틀 후로 변경해줘']

    for inp in input_list:
        nlu_result = prompt_agent.run_nlu_chain(inp=inp)
        user_action = nlu_result['action']

        if user_action == 'read':
            schedule_db = schedule_list
        else:
            schedule_db = []

        dialog_state = prompt_agent.update_dialog_state(dialog_state=dialog_state, db=schedule_db)
        print(f'dialog_state: {dialog_state}')

        dst_result = prompt_agent.run_dst_chain(dialog_state=dialog_state, nlu_result=nlu_result)
        dialog_state = dst_result

        print(f'dialog_state: {dialog_state}')

        dialog_state = prompt_agent.update_dialog_state(dialog_state=dialog_state)
        print(f'updated dialog_state: {dialog_state}')

        system_state = prompt_agent.run_dp_chain(dialog_state=dialog_state)
        print(f'system_state: {system_state}')

        system_state = prompt_agent.update_system_state(system_state=system_state, dialog_state=dialog_state)
        print(f'updated system_state: {system_state}')

        nlg_result = prompt_agent.run_nlg_chain(system_state)
        print(nlg_result)


if __name__ == '__main__':
    main()
