# 欠損値を0で埋める
# 欠損値を平均で埋める
# 欠損値を中央値で埋める
import pandas as pd
from const import INPUT_DIR
from langchain.tools import BaseTool
from utils import text_processing


class Fill0Tool(BaseTool):
    name = "fill_0_tool"
    description = """It is useful to fill in missing values with zeros when a column name is received. The input should be a column name."""

    def _run(self, query: str) -> str:
        """Use the tool."""
        query = text_processing(query)

        df = pd.read_csv(f"{INPUT_DIR}/stored_df.csv")
        df[query] = df[query].fillna(0)
        df.to_csv(f"{INPUT_DIR}/stored_df.csv", index=False)

        result = f"{self.name} The missing values {query} could be filled with 0."
        return result

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")


class FillMeansTool(BaseTool):
    name = "fill_means_tool"
    description = """It is useful to fill in missing values with means when a column name is received. The input should be a column name."""

    def _run(self, query: str) -> str:
        """Use the tool."""
        query = text_processing(query)

        df = pd.read_csv(f"{INPUT_DIR}/stored_df.csv")
        df[query] = df[query].fillna(df[query].mean())
        df.to_csv(f"{INPUT_DIR}/stored_df.csv", index=False)

        result = f"{self.name} The missing values {query} could be filled with means."
        return result

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")


class FillMedianTool(BaseTool):
    name = "fill_median_tool"
    description = """It is useful to fill in missing values with median when a column name is received. The input should be a column name."""

    def _run(self, query: str) -> str:
        """Use the tool."""
        query = text_processing(query)

        df = pd.read_csv(f"{INPUT_DIR}/stored_df.csv")
        df[query] = df[query].fillna(df[query].median())
        df.to_csv(f"{INPUT_DIR}/stored_df.csv", index=False)

        result = f"{self.name} The missing values {query} could be filled with median."
        return result

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")
