
# to run this: streamlit run <filename>
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)
from pydantic import BaseModel, Field, validator

load_dotenv()


class QuestionAndAnswer(BaseModel):
    question: str = Field(description="a question about the text")
    answer: str = Field(description="an answer to the question")

    @validator("question")
    def question_ends_with_question_mark(cls, field):
        if field[-1] != "?":
            raise ValueError("Badly formed question!")
        return field


class QuestionSet(BaseModel):
    translated_text: str = Field("a translated version of the give text")
    questions: list[QuestionAndAnswer]


from langchain.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=QuestionSet)


def create_question_set(text, input_language, output_language) -> QuestionSet:
    qna_prompt = SystemMessagePromptTemplate.from_template(
        """
        You are to produce a few question to test the reader's understanding of a text.

        Steps:
        0. Translate the text from {input_language} into {output_language}
        1. Analyse the Text
        2. Summarise it
        3. Produce 1 to 5 questions about the text
        4. Produce an answer to each question

        The questions and answers must be in {output_language}

        {format_instructions}
        
        Text:
        {text}
        """,
        input_variables=["text", "input_language", "output_language"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chat_prompt = ChatPromptTemplate.from_messages([qna_prompt])
    chat = ChatOpenAI(temperature=0)
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    qna_raw = chain.run(
        text=text,
        input_language=input_language,
        output_language=output_language,
        format_instructions=parser.get_format_instructions(),
    )
    return parser.parse(qna_raw)


st.title("Learn a new language by reading stuff")


def generate_quiz():
    with tab2, st.form("quiz_form"):
        text = st.session_state['target_text']
        from_lang = 'english'
        to_lang = 'french'
        qna: QuestionSet = create_question_set(text, from_lang, to_lang)
        st.markdown(qna.translated_text)
        for i, q in enumerate(qna.questions):
            st.text_area(f"Question {i+1}: {q.question}", key=f"response_{i}")
        st.form_submit_button(label="Check my answers", on_click=check)


def check():
    with tab2:
        st.write('All CORRECT')


tab1, tab2 = st.tabs(["Text", "Quiz"])

with tab1, st.form("config_form"):
    st.text_area("You want to learn about this:", key="target_text")
    st.form_submit_button(label="Go", on_click=generate_quiz)

