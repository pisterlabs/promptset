import os

import chainlit as cl
from tabulate import tabulate

import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI


class TableQAFunction:
    @classmethod
    async def run(self, tables=[], question="", new_table_name=""):
        llm = OpenAI(api_token=os.environ.get("OPENAI_API_KEY"))

        pandas_ai = PandasAI(llm)

        dfs = []
        for df in tables:
            dfs.append(cl.user_session.get(df))

        response = pandas_ai(dfs, prompt=question)

        if isinstance(response, (pd.DataFrame, pd.Series)):
            await cl.Text(
                display="page",
                language="json",
                name=f"{new_table_name}_show",
                content=tabulate(
                    response,
                    headers="keys",
                    tablefmt="rounded_outline",
                ),
            ).send()

            cl.user_session.set(new_table_name, response)

            response = f"Escreva para o usuário que a tabela foi criada e está salva em: '{new_table_name}'. Mostre a tabela escrevendo: '{new_table_name}_show.'"

        return str(response)

    @classmethod
    def get_infos(self):
        infos = {
            "name": "table_qa",
            "description": "Ask questions about dataframes, tables and CSVs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tables": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "Tables name.",
                        },
                    },
                    "question": {
                        "type": "string",
                        "description": "Question you want to ask to table.",
                    },
                    "new_table_name": {
                        "type": "string",
                        "description": "Return a unique name for new table.",
                    },
                },
                "required": ["tables", "question", "new_table_name"],
            },
        }

        return infos
