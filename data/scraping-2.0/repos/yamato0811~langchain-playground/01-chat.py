from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import settings

OPEN_AI_API_KEY = settings.OPEN_AI_API_KEY

def main():
    chat_model = ChatOpenAI(openai_api_key=OPEN_AI_API_KEY, model_name="gpt-3.5-turbo")

    text = "What would be a good company name for a company that makes colorful socks?"
    messages = [HumanMessage(content=text)]

    print(chat_model.invoke(messages))


if __name__ == "__main__":
    main()