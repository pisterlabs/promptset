from domino.base_piece import BasePiece
from .models import InputModel, OutputModel, SecretsModel
from typing import List
from enum import Enum
from openai import OpenAI
import tiktoken
import asyncio


class TokenLimit(int, Enum):
    gpt_3_5_turbo = 4000
    gpt_4 = 8000


class TextSummarizerPiece(BasePiece):

    async def chat_completion_method(self, input_data: InputModel, prompt: str, client: OpenAI):
        self.logger.info("Running OpenAI completion request...")
        try:
            response = client.chat.completions.create(
                model = input_data.openai_model,
                messages = [
                    {"role": "user", "content": prompt}
                ],
                temperature = input_data.temperature,
                max_tokens = input_data.completion_max_tokens,
            )
            string_generated_text = response.choices[0].message.content
        except Exception as e:
            self.logger.info(f"\nCompletion task failed: {e}")
            raise Exception(f"Completion task failed: {e}")
        return string_generated_text

    async def agenerate_chat_completion(self, input_data: InputModel, texts_chunks: List, client: OpenAI):
        tasks = [self.chat_completion_method(input_data=input_data, prompt=text, client=client) for text in texts_chunks]
        return await asyncio.gather(*tasks)

    def create_chunks_with_prompt(self, input_data: InputModel, text: str):
        text_chunk_size = input_data.chunk_size - len(self.encoding.encode(text=self.prompt))
        total_text_tokens = self.encoding.encode(text=text)
        chunk_overlap = round(input_data.chunk_overlap_rate * text_chunk_size)
        text_chunks_with_prompt = []
        for i in range(0, len(total_text_tokens), text_chunk_size):
            idx_chunk_start = [i - chunk_overlap if i>0 else 0][0]
            decoded_text_chunk = self.encoding.decode(total_text_tokens[idx_chunk_start:i+text_chunk_size])
            chunk_with_prompt = self.prompt.format(text=decoded_text_chunk)
            text_chunks_with_prompt.append(chunk_with_prompt)
        return text_chunks_with_prompt

    def format_display_result(self, input_data: InputModel, final_summary: str):
        md_text = f"""
## Summarized text
{final_summary}

## Args
**model**: {input_data.openai_model}
**temperature**: {input_data.temperature}
**max_tokens**: {input_data.completion_max_tokens}

"""
        file_path = f"{self.results_path}/display_result.md"
        with open(file_path, "w") as f:
            f.write(md_text)
        self.display_result = {
            "file_type": "md",
            "file_path": file_path
        }

    def piece_function(self, input_data: InputModel, secrets_data: SecretsModel):
        # OpenAI settings
        if secrets_data.OPENAI_API_KEY is None:
            raise Exception("OPENAI_API_KEY not found in ENV vars. Please add it to the secrets section of the Piece.")

        client = OpenAI(api_key=secrets_data.OPENAI_API_KEY)

        # Input arguments
        token_limits = TokenLimit[input_data.openai_model.name].value
        completion_max_tokens = input_data.completion_max_tokens
        text_token_count = token_limits
        if input_data.text_file_path:
             with open(input_data.text_file_path, "r") as f:
                text = f.read()
        else:
            text = input_data.text

        self.prompt = """Write a concise summary of the text below, while maintaining its original writing form.
---
text:

{text}
---
concise summary:"""

        # Summarizing loop
        loop = asyncio.new_event_loop()
        self.encoding = tiktoken.encoding_for_model(input_data.openai_model)
        self.logger.info(f"Loading text")
        while text_token_count > (token_limits - completion_max_tokens):
            texts_chunks_with_prompt = self.create_chunks_with_prompt(input_data=input_data, text=text)
            summaries_chunks = loop.run_until_complete(self.agenerate_chat_completion(input_data, texts_chunks_with_prompt, client))
            text = " ".join(summaries_chunks)
            text_token_count = len(self.encoding.encode(text=text))

        self.logger.info(f"Summarizing text")
        response = loop.run_until_complete(self.agenerate_chat_completion(input_data, [text], client))
        final_summary = response[0]

        # Display result in the Domino GUI
        self.format_display_result(input_data,final_summary)

        if input_data.output_type == "string":
            self.logger.info(f"Returning final summary as a string")
            return OutputModel(
                string_summarized_text=final_summary,
            )

        output_file_path = f"{self.results_path}/summarized_text.txt"
        with open(output_file_path, "w") as f:
            f.write(final_summary)

        if input_data.output_type == "file":
            self.logger.info(f"Saved final summary as file in {output_file_path}")
            return OutputModel(
                file_path_summarized_text=output_file_path
            )

        self.logger.info(f"Returning final summary as a string and file in: {output_file_path}")
        return OutputModel(
            string_summarized_text=final_summary,
            file_path_summarized_text=output_file_path
        )

