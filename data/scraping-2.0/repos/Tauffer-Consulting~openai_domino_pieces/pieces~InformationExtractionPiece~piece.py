from domino.base_piece import BasePiece
from .models import InputModel, OutputModel, SecretsModel
from openai import OpenAI
import json


class InformationExtractionPiece(BasePiece):

    def piece_function(self, input_data: InputModel, secrets_data: SecretsModel):
        # OpenAI settings
        if secrets_data.OPENAI_API_KEY is None:
            raise Exception("OPENAI_API_KEY not found in ENV vars. Please add it to the secrets section of the Piece.")

        client = OpenAI(api_key=secrets_data.OPENAI_API_KEY)
        prompt = f"""Extract the following information from the text below as JSON.
Use the items to be extract as information to identify the right information to be extract:
---
Input text: {input_data.input_text}
Items to be extracted:
{input_data.extract_items}
"""
        response = client.chat.completions.create(
            response_format={
                "type": "json_object"
            },
            temperature=0,
            model=input_data.openai_model,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        if not response.choices:
            raise Exception("No response from OpenAI")

        if response.choices[0].message.content is None:
            raise Exception("No response from OpenAI")

        output_json = json.loads(response.choices[0].message.content)

        # Display result in the Domino GUI
        self.format_display_result(input_data, output_json)

        # Return extracted information
        self.logger.info("Returning extracted information")
        return OutputModel(**output_json)

    def format_display_result(self, input_data: InputModel, result: dict):
        md_text = """## Extracted Information\n"""
        for item in input_data.extract_items:
            md_text += f"""### {item.name}:\n{result.get(item.name)}\n"""
        file_path = f"{self.results_path}/display_result.md"
        with open(file_path, "w") as f:
            f.write(md_text)
        self.display_result = {
            "file_type": "md",
            "file_path": file_path
        }
