import pandas as pd

from sklearn.model_selection import train_test_split

from langchain.tools import BaseTool

class MakeDatasetTool(BaseTool):
    name = "make_dataset_tool"
    description = """It is useful to create train.csv and eval.csv. The column name of the target variable is used as input."""

    def _run(self, query: str) -> str:
        """Use the tool."""

        df = pd.read_csv('/home/langchain-tools/data/sample2.csv')
        X = df.drop(query, axis=1)
        y = df[query]

        # 学習データと評価データに分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=3655)

        X_train[query] = y_train
        X_test[query] = y_test
        
        X_train.to_csv('/home/langchain-tools/data/train.csv', index=False)
        X_test.to_csv('/home/langchain-tools/data/test.csv', index=False)

        result = "train.csv and eval.csv have been created." 
        return result

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")