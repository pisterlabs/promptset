from langchain.chat_models import ChatOpenAI, ChatVertexAI
from langchain.prompts.chat import ChatPromptTemplate

from openai_service.config.config import Settings, get_settings
from openai_service.config.parsers import StringOutputParser


def translate(settings: Settings = get_settings()):
    model = ChatOpenAI(
        model="gpt-3.5-turbo-1106", api_key=settings.openai_api_key, temperature=0.5
    )
    human_template = "{text}"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", settings.translate_prompt),
            ("human", human_template),
        ]
    )

    parser = StringOutputParser()

    return prompt | model | parser


# def translate_gemini(settings: Settings = get_settings()):
#     model = ChatVertexAI(
#         model_name="gemini-pro",
#         top_p=1,
#         temperature=0.9,
#         max_output_tokens=2048,
#     )

#     prompt = ChatPromptTemplate.from_template(settings.gemini_prompt)

#     parser = StringOutputParser()

#     return prompt | model | parser
