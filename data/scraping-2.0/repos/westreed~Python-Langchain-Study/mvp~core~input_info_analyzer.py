from langchain import LLMChain
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)

from mvp.data_manager import *
from mvp.util import remove_indent
from typing import *


def input_info_analyzer(
    data_manager: DataManager,
    evaluation_manager: EvaluationManager
):
    chat_manager = ChatManager()

    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                remove_indent(
                    f"""You are an interviewer at {data_manager.company}.

                    {data_manager.get_userdata()}
                    """)),

            HumanMessagePromptTemplate.from_template(
                remove_indent(
                    """As an interviewer, Please read the cover letter and self-introduction of the interviewee and provide an evaluation from the {company}'s perspective, dividing it into positive aspects and areas for improvement. Please write in Korean. The format is as follows:

                    "좋은점":
                    - Content of positive aspects
                    "아쉬운점":
                    - Content of areas for improvement

                    Please write in Korean.
                    """))
        ],
        input_variables=["company"],
    )

    create_question_chain = LLMChain(llm=chat_manager.get_chat_model(),
                                     prompt=prompt)
    output = create_question_chain(data_manager.company)
    evaluation_manager.add_coverletter_evaluation(output)
