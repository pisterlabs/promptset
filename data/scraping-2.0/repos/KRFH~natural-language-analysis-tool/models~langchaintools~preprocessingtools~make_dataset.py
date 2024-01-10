import pandas as pd
from const import INPUT_DIR, TEST_SIZE, RANDOM_STATE
from sklearn.model_selection import train_test_split
from utils import text_processing
from langchain.tools import BaseTool


class MakeDatasetTool(BaseTool):
    name = "make_dataset_tool"
    description = """It is useful to create train.csv and eval.csv. The column name of the target variable should be used as input."""

    def _run(self, query: str) -> str:
        """Use the tool."""
        query = text_processing(query)

        df = pd.read_csv(f"{INPUT_DIR}/stored_df.csv")
        X = df.drop(query, axis=1)
        y = df[query]

        # 学習データと評価データに分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        X_train[query] = y_train
        X_test[query] = y_test

        X_train.to_csv(f"{INPUT_DIR}/train.csv", index=False)
        X_test.to_csv(f"{INPUT_DIR}/test.csv", index=False)
        # y_train.to_csv(
        #     f"{INPUT_DIR}/train_target.csv", index=False
        # )
        # y_test.to_csv(
        #     f"{INPUT_DIR}/test_target.csv", index=False
        # )

        dataset_info = {
            "Number of Training Examples ": X_train.shape[0],
            "Number of Test Examples ": X_test.shape[0],
            "Training X Shape ": X_train.shape,
            "Training y Shape ": y_train.shape[0],
            "Test X Shape ": X_test.shape,
            "Test y Shape ": y_test.shape[0],
            "train columns": X_train.columns,
            "test columns": X_test.columns,
        }

        result = f"{self.name} train.csv and eval.csv have been created.{dataset_info}"
        return result

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")
