import pynecone as pc
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    SystemMessage
)

import os
key = open('../api-key', 'r').readline()
os.environ["OPENAI_API_KEY"] = key

chat = ChatOpenAI(temperature=0.8)
system_message = "assistant는 마케팅 문구 작성 도우미로 동작한다. user의 내용을 참고하여 마케팅 문구를 작성해라"
system_message_prompt = SystemMessage(content=system_message)

human_template = ("제품 이름: {product_name}\n"
                  "제품 설명: {product_desc}\n"
                  "제품 톤앤매너: {product_tone_and_mannar}\n"
                  "위 정보를 참조해서 마케팅 문구 만들어줘"
                  )

human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

chain = LLMChain(llm=chat, prompt=chat_prompt)


class State(pc.State):
    """The app state."""

    product_name: str = ""
    product_desc: str = ""
    product_tone_and_mannar: str = ""

    content: str = ""

    is_working: bool = False

    async def handle_submit(self, form_data):
        self.is_working = True
        self.product_name = form_data['product_name']
        self.product_desc = form_data['product_desc']
        self.product_tone_and_mannar = form_data['product_tone_and_mannar']

        ad_slogan_list = []
        for i in range(10):
            ad_slogan = chain.run(product_name=self.product_name,
                            product_desc=self.product_desc,
                            product_tone_and_mannar=self.product_tone_and_mannar)
            ad_slogan_list.append(f"- {ad_slogan}")

            self.content = "\n".join(ad_slogan_list)
            yield

        self.content = "\n".join(ad_slogan_list)

        self.is_working = False


def index() -> pc.Component:
    return pc.center(
        pc.vstack(
        pc.form(
        pc.vstack(
            pc.heading("콘텐츠 마케팅 AI 서비스", font_size="2em"),
            pc.text("제품 이름"),
            pc.input(placeholder="제품 이름", id="product_name"),

            pc.text("주요 내용"),
            pc.input(placeholder="주요 내용", id="product_desc"),

            pc.text("광고 문구 톤앤 매너"),
            pc.select(["신뢰", "유쾌", "엉뚱"], id="product_tone_and_mannar"),

            pc.button("Submit", type_="submit"),

        ),
        on_submit=State.handle_submit,

        width="100%",
        ),
        pc.cond(State.is_working,
                pc.spinner(
                    color="lightgreen",
                    thickness=5,
                    speed="1.5s",
                    size="xl",
                ),),
        pc.box(pc.markdown(State.content)),

        spacing="1em",
        font_size="1em",
        width="80%",

        padding_top="10%",)
    )



# Add state and page to the app.
app = pc.App(state=State)
app.add_page(index)
app.compile()
