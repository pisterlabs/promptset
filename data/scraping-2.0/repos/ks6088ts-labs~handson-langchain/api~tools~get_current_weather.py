# OpenAI APIとLangChainエージェントでFunction Calling機能を使用する ref. https://zenn.dev/onikarubi/articles/bbb235defa740b
import json
from enum import Enum
from typing import Optional, Type

from langchain.agents.tools import BaseTool
from pydantic import BaseModel, Field


# 温度の単位を表すEnumクラスを定義します
class WeatherUnit(str, Enum):
    celsius = "celsius"
    fahrenheit = "fahrenheit"


# 天気情報を取得するための入力パラメータを定義するモデルを作成します
class GetCurrentWeatherCheckInput(BaseModel):
    location: str = Field(..., description="city_and_state")
    unit: WeatherUnit


# 天気情報を取得する機能を定義
class GetCurrentWeatherTool(BaseTool):
    name = "get_current_weather"
    description = "Acquire current weather at a specified location."

    # _runメソッドに具体的な機能の実装を記述
    def _run(self, location: str, unit: str = "fahrenheit"):
        # 実際は外部の天気API等から情報を取得
        weather_info = {
            "location": location,
            "temperature": "30",
            "unit": unit,
            "forecast": ["sunny", "windy"],
        }
        return json.dumps(weather_info)

    def _arun(self, location: str, unit: str):
        raise NotImplementedError("This tool does not support async")

    # 入力パラメータのスキーマを定義
    args_schema: Optional[Type[BaseModel]] = GetCurrentWeatherCheckInput
