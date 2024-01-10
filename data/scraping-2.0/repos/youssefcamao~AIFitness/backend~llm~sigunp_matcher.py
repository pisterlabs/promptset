from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from ..models.llm_models import ErrorMessage, MatchedSignup


class SignupMatcher:
    def __init__(self):
        self.template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        "Your role is to assist users in creating a unique signup process. Users will provide text containing their username, email, and password, along with a 'login secret' – a unique phrase or word they choose for enhanced security"
                        "Your task is to parse this information and return a JSON object containing the email, password, and full name"
                        "Additionally, you'll create a security question based on their login secret, forming an object with a question and response"
                        "If any information is missing or incorrect, you will generate a short error message in JSON format. Your responses should always be in JSON, focusing on clarity and correctness"
                        "In the event of any missing or incorrect information, you are to generate an error message in the specific format: {'error': 'output the error'}"
                        "When the signup details are successfully processed, your response should be in a JSON format resembling the following example:{'full_name': 'youssef sbai','email': 'youssef@gmail.com','password': '123youssef','security': {'question': 'What is your dad's favorite TV show?','response': 'power rangers'}"
                    )
                ),
                HumanMessagePromptTemplate.from_template("{text}"),
            ]
        )
        optional_params = {"response_format": {"type": "json_object"}}
        self.chat_model = ChatOpenAI(
            model_name="gpt-4-1106-preview", model_kwargs=optional_params)

    async def match_sigunup_text(self, text: str) -> ErrorMessage | MatchedSignup:
        model_ouput = await self.chat_model.ainvoke(
            self.template.format_messages(text=text))
        response = model_ouput.content
        if "full_name" in response and "email" in response and "password" in response and "security" in response:
            return MatchedSignup.from_json(response)
        elif "error" in response:
            return ErrorMessage.from_json(response)
        else:
            raise ValueError("Invalid JSON format")
