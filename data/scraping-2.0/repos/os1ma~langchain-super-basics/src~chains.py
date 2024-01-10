from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()

prompt = PromptTemplate.from_template(
    """以下の料理のレシピを考えてください。
                                      
料理名: {dish}"""
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

chain = prompt | llm

result = chain.invoke({"dish": "カレー"})
print(result.content)
