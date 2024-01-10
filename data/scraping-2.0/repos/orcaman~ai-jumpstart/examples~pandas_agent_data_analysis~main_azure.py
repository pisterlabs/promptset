import sys
import dotenv

dotenv.load_dotenv()

from langchain.llms import AzureOpenAI
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd


class CSVAnalyzer:
    def __init__(self, data_file_path, query):
        self.data_file_path = data_file_path
        self.query = query
        self.agent = None
        self.df = None

    def load_df(self):
        self.df = pd.read_csv(self.data_file_path)

    def load_agent(self):
        self.agent = create_pandas_dataframe_agent(AzureOpenAI(deployment_name='text-davinci-003', model_name="text-davinci-003"), self.df, verbose=True)

    def analyze_csv(self):
        self.load_df()
        self.load_agent()

        print('\n\n\n\n\n-----------------')
        # the query might contain several questions, separated by a semicolon. We split them and run them one by one
        for q in self.query.split(';'):
            print('Question:', q)
            self.agent.run(q)
        print('-----------------\n\n')


def main():
    if len(sys.argv) < 3:
        print("Missing arguments: data file and/or query")
        sys.exit(1)

    data_file_path = sys.argv[1]
    query = sys.argv[2]

    analyzer = CSVAnalyzer(data_file_path, query)
    analyzer.analyze_csv()


if __name__ == "__main__":
    main()
