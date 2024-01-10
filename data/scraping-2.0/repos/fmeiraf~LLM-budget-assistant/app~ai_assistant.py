from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import openai
from pandasai.helpers.anonymizer import anonymize_dataframe_head
import re
import json
import pandas as pd

import dotenv
import os

dotenv.load_dotenv()
OPEN_API_KEY = os.getenv("OPENAI_API_KEY")

MODEL_INSTRUCTION = "You are an expert in responding financial questions. Never provide fake information. Output only valid JSON."

TRIAGE_PROMPT = """
You will be given a financial question and the following information:
    - dataframe with the following columns: {df_columns}
    - spending categories: {spending_categories}


Your goal is to analyze this question and provide the following object as output:


    is_valid: if the question can be answered by querying the dataframe return True, otherwise False.
    question: if is_valid=True, return the question as is, otherwise request the user to rephrase the question.
    objective_question: if is_valid=True, outline the question in a way that can be answered by querying the dataframe,
    remember to use the columns and spending categories provided to you when needed.
    answer: if is_valid=False, provide your a followup answer to help the user rephrase the question, otherwise return None.


question: {question}
"""

PANDAS_AI_BASE_PROMPT = "{question}"

PANDAS_AI_PLOT_PROMPT = """create a plot to answer this question: {question}.
                            If you can't create a plot, return None.
                        """


class AIAssistant:
    def __init__(
        self,
        model: str,
        model_instruction: str = MODEL_INSTRUCTION,
        save_chat_history: bool = True,
    ):
        self.model = model
        self.llm = OpenAI(api_token=OPEN_API_KEY)
        self.pandas_ai = PandasAI(
            llm=self.llm,
            conversational=True,
            save_charts=True,
            save_charts_path="charts",
            enable_cache=False,
        )
        self.chat_history = []
        self.save_chat_history = save_chat_history

        if model_instruction is None:
            self.model_instruction = "You are a helpful assistant."
        else:
            self.model_instruction = model_instruction

        self.chat_history.append({"role": "system", "content": self.model_instruction})

    def ask(self, question: str):
        self.chat_history.append({"role": "user", "content": question})
        response = openai.ChatCompletion.create(
            model=self.model, messages=self.chat_history, temperature=0
        )

        if self.save_chat_history:
            self.chat_history.append(
                {
                    "role": "system",
                    "content": response["choices"][0]["message"]["content"],
                }
            )

        return json.loads(response["choices"][0]["message"]["content"])

    def ask_with_pandasai(self, dataframe: pd.DataFrame, question: str):
        text = self.pandas_ai(
            dataframe, prompt=PANDAS_AI_BASE_PROMPT.format(question=question)
        )

        self.pandas_ai(
            dataframe, prompt=PANDAS_AI_PLOT_PROMPT.format(question=question)
        )

        plot_path = self.get_last_plot_created()

        if self.save_chat_history:
            self.chat_history.append({"role": "system", "content": text})

        return text, plot_path

    def run_triage_prompt(self, dataframe: pd.DataFrame, question: str):
        df_ref = {
            "df_columns": ", ".join(anonymize_dataframe_head(dataframe).columns),
            "spending_categories": ", ".join(dataframe["category"].unique()),
        }

        prompt = TRIAGE_PROMPT.format(**df_ref, question=question)

        response = self.ask(question=prompt)
        print(response)

        if response:
            if response["is_valid"]:
                answer, plot_path = self.ask_with_pandasai(
                    dataframe, response["objective_question"]
                )
                return {
                    "is_valid": True,
                    "question": question,
                    "answer": answer,
                    "plot_path": plot_path,
                }
            else:
                return {
                    "is_valid": False,
                    "question": question,
                    "answer": response["answer"],
                    "plot_path": None,
                }
        else:
            raise ValueError("LLM answer does not evaluate into valid object")

    def chat(self, dataframe: pd.DataFrame, question: str):
        response = self.run_triage_prompt(dataframe, question)
        return response

    def extract_plot_path_from_call(self, code_call: str):
        match = re.search(r"savefig\('(.*?)'\)", code_call)
        if match:
            return match.group(1)
        else:
            return None

    def get_last_plot_created(self):
        last_plot_path = self.extract_plot_path_from_call(
            self.pandas_ai.last_code_executed
        )
        return last_plot_path
