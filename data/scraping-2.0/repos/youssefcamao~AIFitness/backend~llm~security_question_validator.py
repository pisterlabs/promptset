from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from ..models.chat_session import UserSecurityQuestion
from ..models.llm_models import ValidationResponse


class SecurityQuestionValidator:
    def __init__(self):
        self.template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        "Your role is to evaluate security questions and answers"
                        "You will receive a security question and its correct answer, along with the user's response"
                        "Your goal is to compare the user's response with the correct answer"
                        "If they match in meaning, regardless of language or expression, you will confirm the match"
                        "Your response should be a JSON object, strictly formatted as {'isResponseValid' : true} for a match, or {'isResponseValid' : false} otherwise. Emphasize understanding the intent and information in the responses rather than exact wording"
                        "Avoid providing additional commentary or information"
                    )
                ),
                HumanMessagePromptTemplate.from_template(
                    "security question: {security_question}\ncorrect answer: {correct_answer}\nuser response: {user_response}"),
            ]
        )
        optional_params = {"response_format": {"type": "json_object"}}
        self.chat_model = ChatOpenAI(
            model_name="gpt-4-1106-preview", model_kwargs=optional_params)

    async def validate_security_question(self, security_question_obj: UserSecurityQuestion, user_response: str) -> bool:
        model_ouput = await self.chat_model.ainvoke(
            self.template.format_messages(security_question=security_question_obj.question, correct_answer=security_question_obj.response, user_response=user_response))
        response = model_ouput.content
        return ValidationResponse.from_json(response)
