import asyncio

from langstream import StreamOutput, collect_final_output
import asyncio
from typing import AsyncGenerator


def function_call_weather_example() -> AsyncGenerator[StreamOutput[str], None]:
    from typing import TypedDict, Literal

    class WeatherReturn(TypedDict):
        location: str
        forecast: str
        temperature: str

    def get_current_weather(
            location: str, format: Literal["celsius", "fahrenheit"] = "celsius"
    ) -> WeatherReturn:
        return WeatherReturn(
            location=location,
            forecast="sunny",
            temperature="25 C" if format == "celsius" else "77 F",
        )

    from typing import Union
    from langstream import Stream
    from langstream.contrib import OpenAIChatStream, OpenAIChatMessage, OpenAIChatDelta
    import json

    stream: Stream[str, Union[OpenAIChatDelta, WeatherReturn]] = OpenAIChatStream[
        str, OpenAIChatDelta
    ](
        "WeatherStream",
        lambda user_input: [
            OpenAIChatMessage(role="user", content=user_input),
        ],
        model="gpt-3.5-turbo",
        functions=[
            {
                "name": "get_current_weather",
                "description": "Gets the current weather in a given location, use this function for any questions related to the weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "description": "The city to get the weather, e.g. San Francisco. Guess the location from user messages",
                            "type": "string",
                        },
                        "format": {
                            "description": "A string with the full content of what the given role said",
                            "type": "string",
                            "enum": ("celsius", "fahrenheit"),
                        },
                    },
                    "required": ["location"],
                },
            }
        ],
        temperature=0,
    ).map(
        lambda delta: get_current_weather(**json.loads(delta.content))
        if delta.role == "function" and delta.name == "get_current_weather"
        else delta
    )

    return stream("What is the weather in Stockholm?")


async def main():
    print("\n\nRun function_call_weather_example:")
    function_call_weather_example_result = await collect_final_output(function_call_weather_example())
    print(function_call_weather_example_result)

    print("\nAll done!")


if __name__ == "__main__":
    asyncio.run(main())
