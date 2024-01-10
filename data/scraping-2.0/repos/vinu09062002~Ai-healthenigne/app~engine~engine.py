import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI

from app.config.config import OPENAI_API_TOKEN

class HealthEngine:
    def __init__(self, dataframe: pd.DataFrame):
        self.api_token = OPENAI_API_TOKEN
        self.llm = OpenAI(api_token=self.api_token)
        self.pandas_ai = PandasAI(self.llm, conversational=False)
        self.dataframe = dataframe

    async def run(self,prompt: str):
        return self.pandas_ai(self.dataframe, prompt)
