import os
import time

from pandasai import PandasAI
from pandasai.llm.openai import OpenAI

import chainlit as cl


class ChartFunction:
    @classmethod
    async def run(self, dataframes=[], description="", chart_name=""):
        llm = OpenAI(api_token=os.environ.get("OPENAI_API_KEY"))

        charts_path = os.getcwd()
        pandas_ai = PandasAI(llm, save_charts=True, save_charts_path=charts_path)

        dfs = []
        for df in dataframes:
            dfs.append(cl.user_session.get(df))

        await cl.make_async(pandas_ai)(dfs, prompt=description)

        time.sleep(5)

        created_chart_path = self._get_latest_created_chart_path(charts_path)

        await cl.Image(
            path=created_chart_path, name=chart_name, display="inline"
        ).send()

        response = f"Escreava para o usuário que o gráfico foi criado e mostre ele para o usuário escrevendo: '{chart_name}'."

        return response

    @classmethod
    def get_infos(self):
        infos = {
            "name": "chart",
            "description": "Create charts from dataframes, tables and CSVs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataframes": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "Dataframe name.",
                        },
                    },
                    "description": {
                        "type": "string",
                        "description": "Describe the chart you want to create.",
                    },
                    "chart_name": {
                        "type": "string",
                        "description": "Give a unique name to the chart.",
                    },
                },
                "required": ["dataframes", "description", "chart_name"],
            },
        }

        return infos

    def _get_latest_created_chart_path(charts_path):
        charts_dir = os.path.join(charts_path, "exports", "charts")

        subdirectories = [
            subd
            for subd in os.listdir(charts_dir)
            if os.path.isdir(os.path.join(charts_dir, subd))
        ]

        sorted_subdirectories = sorted(
            subdirectories,
            key=lambda subdir: os.path.getmtime(os.path.join(charts_dir, subdir)),
            reverse=True,
        )

        latest_dir = sorted_subdirectories[0] if sorted_subdirectories else None

        chart_path = os.path.join(charts_dir, latest_dir, "chart.png")

        return chart_path
