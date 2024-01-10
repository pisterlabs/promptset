from gptfunctionutil import GPTFunctionLibrary, AILibFunction, LibParam
import openai
import asyncio

"""

"""


class MyLib(GPTFunctionLibrary):
    @AILibFunction(name="wait_for", description="Wait for a few seconds, then return.")
    @LibParam(targetuser="Number of seconds to wait for.")
    async def wait_for(self, towait: int):
        # Wait for a set period of time.
        print("launcing waitfor.")
        await asyncio.sleep(towait)
        return f"waited for {towait}'!"


async def main():
    # Initialize your subclass before calling the API.
    client = openai.AsyncClient()
    mylib = MyLib()

    # Call OpenAI's api
    completion = await client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Wait for 25 seconds."},
        ],
        tools=mylib.get_tool_schema(),
        tool_choice="auto",
    )
    message = completion.choices[0].message
    if message.tool_calls:
        for tool in message.tool_calls:
            result = await mylib.call_by_tool_async(tool)
            # Print result
            print(result)
    else:
        # Unable to tell that it's a function.
        print(completion.choices[0].message.content)


asyncio.run(main())
