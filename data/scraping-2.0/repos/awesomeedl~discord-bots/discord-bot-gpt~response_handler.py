import openai
import os

openai.api_key = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL')

char_limit = 2000

async def handle_response(message: str, chat_history: list[dict[str, str]]) -> list[str]:
    try:
        completion = await openai.ChatCompletion.acreate(
            model=OPENAI_MODEL,
            messages=chat_history + [{"role": "user", "content": message}])
    except Exception:
        raise

    return split_string(completion.choices[0].message.content)


def split_string(s: str):
    chunks = []

    lines = s.split('\n')  # Split into lines

    buffer, buffer_length = [], 0
    code_block_open = False

    for line in lines:
        line_length = len(line)

        if buffer_length + line_length > char_limit:
            if code_block_open:
                chunks.append('\n'.join(buffer) + "```")
                buffer = ["```"]
            else:
                chunks.append('\n'.join(buffer))
                buffer = []

            buffer_length = 0
            code_block_open = False

        if "```" in line:
            code_block_open = not code_block_open

        buffer.append(line)
        buffer_length += line_length + 1

    if code_block_open:
        chunks.append('\n'.join(buffer) + "```")
    else:
        chunks.append('\n'.join(buffer))

    return chunks


async def handle_image_response(prompt: str, resolution: str) -> str:

    try:
        image = await openai.Image.acreate(
            prompt=prompt,
            n=1,
            size=resolution
        )
    except Exception:
        raise

    return image.data[0].url



#
# # Response is ok, send directly
# if len(response) <= char_limit:
#     await message.reply(response)
#     return
#
# # Response is too long
# # Split the response into smaller chunks of no more than 1900 characters each(Discord limit is 2000 per chunk)
# if "```" not in response:
#     response_chunks = [response[i:i + char_limit] for i in range(0, len(response), char_limit)]
#
#     for chunk in response_chunks:
#         await message.reply(chunk)
#     return
#
# # Code block exists
# parts = response.split("```")
#
# for i in range(0, len(parts)):
#     if i % 2 == 0:  # indices that are even are not code blocks
#
#         await message.reply(parts[i])
#
#     # Send the code block in a separate message
#     else:  # Odd-numbered parts are code blocks
#         code_block = parts[i].split("\n")
#         formatted_code_block = ""
#         for line in code_block:
#             while len(line) > char_limit:
#                 # Split the line at the 50th character
#                 formatted_code_block += line[:char_limit] + "\n"
#                 line = line[char_limit:]
#             formatted_code_block += line + "\n"  # Add the line and separate with new line
#
#         # Send the code block in a separate message
#         if len(formatted_code_block) > char_limit + 100:
#             code_block_chunks = [formatted_code_block[i:i + char_limit]
#                                  for i in range(0, len(formatted_code_block), char_limit)]
#             for chunk in code_block_chunks:
#                 await message.reply("```" + chunk + "```")
#         else:
#             await message.reply("```" + formatted_code_block + "```")
