from . import measure_time
from . import credentials
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd
import os


# os.environ['OPENAI_API_KEY'] =  credentials.API_KEY


@measure_time.measure_time
def ask_agent(dataset: str, prompt: str, sep=",") -> str:
    try:
        df = pd.read_csv(dataset, sep=sep)
    except Exception as e:
        return f"Error: {e}"

    chat = ChatOpenAI(model="gpt-3.5-turbo")
    agent = create_pandas_dataframe_agent(
        llm=chat,
        df=df,
        verbose=True
    )
    response = agent.run(prompt)
    return response


def main():
    dataset = "/home/ziemian/Code/django/sql-gen/sql_gen/example_data/Iris.csv"
    prompt = "How many rows are in the dataset"
    response = ask_agent(dataset, prompt)
    print(f"Response: {response}")


if __name__ == "__main__":
    main()