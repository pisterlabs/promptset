import datetime
from openai import OpenAI


class LLMExtractor:
    """
    This class is responsible for running the FHIR resource extraction using OpenAI's LLM model.
    """

    def __init__(self, config, fhir_tools):
        # copy config
        self.config = config

        # copy functions
        self.fhir_tools = fhir_tools

        # set up openai client
        self.client = OpenAI(api_key=config["OPENAI"]["API_KEY"])

    def extract(self, text):
        """run the LLM model on the text"""

        # create initial conversation
        messages = [
            {
                "role": "system",
                "content": self.config["GENAI"]["SYSTEM_PROMPT"].format(
                    date=datetime.datetime.now().strftime("%Y-%m-%d")
                ),
            },
            {
                "role": "user",
                "content": text,
            },
        ]

        # initial llm request
        response = self.client.chat.completions.create(
            model=self.config["OPENAI"]["MODEL"],
            messages=messages,
            tools=self.fhir_tools.tools,
            tool_choice="auto",
        )
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        # check if the model wanted to call a function
        if tool_calls:
            # extend conversation with assistant's reply
            messages.append(response_message)

            # apply all function calls
            for tool_call in tool_calls:
                # run the function call
                function_response = self.fhir_tools.run(tool_call)

                # extend conversation with function response
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_call.function.name,
                        "content": function_response,
                    }
                )

            # send the conversation back to the model
            second_response = self.client.chat.completions.create(
                model=self.config["OPENAI"]["MODEL"],
                messages=messages,
            )

            return second_response

        else:
            return response
