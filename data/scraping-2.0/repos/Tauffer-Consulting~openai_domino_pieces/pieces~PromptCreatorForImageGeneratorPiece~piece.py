from domino.base_piece import BasePiece
from .models import InputModel, OutputModel, SecretsModel
from openai import OpenAI

class PromptCreatorForImageGeneratorPiece(BasePiece):
    def openai_response(self, input_data: InputModel, prompt: str, client: OpenAI):
        # Input arguments
        openai_model = input_data.openai_model
        completion_max_tokens = input_data.completion_max_tokens
        temperature = input_data.temperature

        try:
            response = client.chat.completions.create(
                model=openai_model,
                messages = [
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=completion_max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.info(f"\nCompletion task failed: {e}")
            raise Exception(f"Completion task failed: {e}")

    def piece_function(self, input_data: InputModel, secrets_data: SecretsModel):
        if secrets_data.OPENAI_API_KEY is None:
            raise Exception("OPENAI_API_KEY not found in ENV vars. Please add it to the secrets section of the Piece.")

        client = OpenAI(api_key=secrets_data.OPENAI_API_KEY)

        template = """You have access to an AI that generates images through text prompts.
Your function is to write a prompt for this AI from a given context. It is very importat that the maximum size of the output prompt must be 1000 characters.
Keep in mind that the AI generating images has no knowledge of the context you've been given. Therefore, it's crucial to include all the important information in the prompt you generate.
You  always write a short prompt that is designed to help the image generator AI create an image for the given context.
You are very good at writing these text prompts for any context that is given  to you.
You're very creative in how you describe the context you've been given, and like to vary the mood that runs through the prompt you've written.
You always suggest some specific art style for the AI to create the image.
For this one the art style would be: {art_style}
Now, create a prompt to help the image generator AI to create an image for this context:
{context}"""

        prompt = template.format(art_style=input_data.art_style, context=input_data.context)
        self.logger.info(f"Generating prompt")
        generated_prompt = self.openai_response(input_data, prompt, client)

        if input_data.output_type == "string":
            self.logger.info("Returning prompt as a string")
            self.format_display_result(input_data, generated_prompt)
            return OutputModel(
                generated_prompt_string=generated_prompt,
            )

        output_file_path = f"{self.results_path}/generated_prompt.txt"
        with open(output_file_path, "w") as f:
            f.write(generated_prompt)

        if input_data.output_type == "file":
            self.logger.info(f"Prompt file saved at: {output_file_path}")
            self.format_display_result(input_data, generated_prompt)
            return OutputModel(
                generated_prompt_file_path=output_file_path
            )

        self.logger.info(f"Returning prompt as a string and file in: {output_file_path}")
        self.format_display_result(input_data, generated_prompt)
        return OutputModel(
            generated_prompt_string=generated_prompt,
            generated_prompt_file_path=output_file_path
        )

    def format_display_result(self, input_data: InputModel, generated_prompt_string: str):
        md_text = f"""
## Generated prompt:
{generated_prompt_string}

## Args
**context**: {input_data.context}
**model**: {input_data.openai_model}
**temperature**: {input_data.temperature}
**completion_max_tokens**: {input_data.completion_max_tokens}
"""
        file_path = f"{self.results_path}/display_result.md"
        with open(file_path, "w") as f:
            f.write(md_text)
        self.display_result = {
            "file_type": "md",
            "file_path": file_path
        }