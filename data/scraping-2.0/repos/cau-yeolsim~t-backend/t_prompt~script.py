import os

from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

# initialize ChatGPT model
chat = ChatOpenAI(
    temperature=0,
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    model_name="gpt-3.5-turbo",
)
system_template = """너는 심리 상담을 위한 챗봇이야.
답변은 무조건 한글로 해야 하고, 존댓말을 사용해야 해.
너는 상대방의 말을 듣고, 관련된 심리학적 지식과 함께 위로를 해줘야해.
아래의 조건들을 지키면서 너는 상대방에게 위로가 되는 말을 해야해.
첫번째 문장에서는 상황을 이해하면서 공감해줘
너의 대답에는 항상 심리학적 사실에 대한 근거가 있어야 해. 맹목적인 공감이 아니라 심리학적 사실에 기반해야 해.
마지막 문장에서는 너가 했던 말들을 종합해서 다시 위로를 해주고, 다음 조언을 위한 질문을 해줘  
"""

system_message_prompt_template = SystemMessagePromptTemplate.from_template(
    system_template
)
human_template = "{sample_text}"
human_message_prompt_template = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt_template = ChatPromptTemplate.from_messages(
    [system_message_prompt_template, human_message_prompt_template]
)
final_prompt = chat_prompt_template.format_prompt(
    output_language="ko",
    max_words=15,
    sample_text="나 요즘 일에 지친 것 같아. 번아웃이 온 것 같아",
).to_messages()
# generate the output by calling ChatGPT model and passing the prompt
completion = chat(final_prompt)
