from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.schema import HumanMessage

 
if __name__ == "__main__":
    load_dotenv()

    chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9)

    text = "What would be a good company name for a company that makes colorful socks?"
    messages = [HumanMessage(content=text)]
    res = chat_model.invoke(messages)

    print(res)



