from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from utils import load_env

if __name__ == "__main__":
    load_env()

    print("Write Quit or Exit to exit.")
    while True:
        query = input("What text do you want to summarize? \n")
        if query == "":
            print("Please provide text to summarize!")
            continue
        elif query.lower() in ["quit", "exit"]:
            print("Quiting application.")
            break
        messages = [
            SystemMessage(content="You are an expert copywriter with expertize in summarizing documents"),
            HumanMessage(content=f"Please provide a short and consice summary of the following text:\n Text{query}"),
        ]
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        summary = llm(messages)
        print(f'SUMMARY \n {"-"*50} \n {summary.content} \n')
