import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import json
class OpenAIHelper:
    """A helper class to interact with OpenAI's API for specific tasks such as extracting information from invoices."""

    def __init__(self, model="gpt-3.5-turbo-1106"):
        """Initializes the OpenAIHelper with the specified model and API key."""
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = AsyncOpenAI(api_key=self.api_key)

    async def extract_fields_from_invoice(self, invoice_text, max_length=16385):
        """
        Asynchronously extracts fields from a given invoice text.

        :param invoice_text: Text of the invoice from which to extract information.
        :param max_length: Maximum length of text to process in one go. Defaults to 4096.
        :return: Extracted invoice details as a string.
        """
        segments = self._split_text(invoice_text, max_length)
        prompt = ("Please extract the following details from the invoice: "
                  "1. Invoice number, invoice date, country of origin, supplier, and total. "
                  "2. For each item in the invoice, list the part number, description, quantity, unit of measure, cost, and weight.")
        json_format = """{
            "invoice_number": "str",
            "invoice_date": "str",
            "country_of_origin": "str",
            "supplier": "str",
            "total": "float",
            "items": [
                {
                "part_number": "str",
                "description": "str",
                "quantity": "int",
                "unit_of_measure": "str",
                "cost": "float",
                "weight": "float"
                }
            ]}"""

        responses = []
        for segment in segments:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an assistant skilled in extracting specific information from structured documents like invoices."},
                        {"role": "system", "content": f"Return data in the following format: JSON with key-value pairs. {json_format}"+ "and respect str and int types remove $ of values."},
                        {"role": "user", "content": f"{prompt}\n\n{segment}"}
                    ],
                )
                responses.append(response.choices[0].message.content)
                if response.choices[0].finish_reason == 'length':
                    responses.append(await self.continue_conversation(response.choices[0], segment))
            except Exception as e:
                print(f"An error occurred: {e}")
                # Additional error handling logic can be added here.

            combined_response = {}
            for response in responses:
                try:
                    #regex for extract json el primer { y el ultimo }
                    response =  response[response.find("{"):response.rfind("}")+1]
                    partial_json = json.loads(response)
                    for key, value in partial_json.items():
                        combined_response.setdefault(key, []).append(value)
                except json.JSONDecodeError:
                    print("Error al decodificar JSON:", response)

            return json.dumps(combined_response, indent=4)

    async def continue_conversation(self, conversation, next_prompt):
        """
        Continues a conversation with OpenAI's model in case of incomplete responses.

        :param conversation: The current conversation context.
        :param next_prompt: The next prompt to continue the conversation.
        :return: The continued part of the conversation.
        """
        try:
            response = await self.client.chat.completions.create(model=self.model, messages=conversation)
            if response.choices[0].finish_reason == 'length':
                conversation.append({'role': 'user', 'content': next_prompt})
                next_response = await self.client.chat.completions.create(model=self.model, messages=conversation)
                return next_response.choices[0].message.content
            return response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred: {e}")
            # Additional error handling logic can be added here.

    def _split_text(self, text, max_length=4096):
        """
        Splits a given text into smaller segments based on the specified maximum length.

        :param text: The text to split.
        :param max_length: The maximum length of each segment.
        :return: A list of text segments.
        """
        segments, segment = [], ""
        for word in text.split():
            if len(segment) + len(word) + 1 < max_length:
                segment += word + " "
            else:
                segments.append(segment)
                segment = word + " "
        segments.append(segment)
        return segments
