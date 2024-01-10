from operator import itemgetter

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

import settings

OPEN_AI_API_KEY = settings.OPEN_AI_API_KEY

prompt1 = ChatPromptTemplate.from_template("what is the city {person} is from?")
prompt2 = ChatPromptTemplate.from_template(
    "what country is the city {city} in? respond in {language}"
)

chat_model = ChatOpenAI(openai_api_key=OPEN_AI_API_KEY, model_name="gpt-3.5-turbo")

chain1 = prompt1 | chat_model | StrOutputParser()

chain2 = (
    {"city": chain1, "language": itemgetter("language")}
    | prompt2
    | chat_model
    | StrOutputParser()
)

output = chain2.invoke({"person": "obama", "language": "日本語"})
print(output)