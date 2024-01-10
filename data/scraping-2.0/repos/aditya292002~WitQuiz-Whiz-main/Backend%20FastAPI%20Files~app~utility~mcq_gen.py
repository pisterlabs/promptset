from icecream import ic
from openai import OpenAI
import asyncio
from ..config import OPENAI_API_KEY, MCQ_GENERATOR_Assistant_id, BLOCK_SIZE, SEND_SIZE, COUNT_BLOCK
import time


client = OpenAI(api_key=OPENAI_API_KEY)

INSTRUCTIONS = """
Assistant "must" format the question as "json", "strictly" following the following rule.
{
    Assistant must generate "two" questions. 
    "question": "Write the question here.",
    "options": {
        "1": "Option 1",
        "2": "Option 2",
        "3": "Option 3",
        "4": "Option 4"
    },
    "answer": "Option(x)",
    "description": "some description"
}
"""

def create_message_run_and_wait(thread_id, assistant_id, instructions, content):
    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content
    )

    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        instructions=instructions
    )

    while True:
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        time.sleep(3)
        print(run.status)
        if run.status in ("completed", "failed"):
            break

    return run


async def get_mcq_each_block(thread_id, assistant_id, some_data):
    ic(some_data)
    run = create_message_run_and_wait(
        thread_id,
        assistant_id,
        INSTRUCTIONS,
        some_data
    )

    messages = client.beta.threads.messages.list(thread_id=thread_id)
    output = messages.data[0].content[0].text.value
    print(output)
    return output

async def generate(data):
    thread = client.beta.threads.create()
    ans = []  # this is the list containing each mcq in json format

    # generating mcq for each block asynchronously
    # to avoid overuse of API (in testing environment)
    # we will process only 2 blocks
    data = data[:min(len(data), BLOCK_SIZE * 3)]
    chunks = []
    ind = 0
    count_chunks = 0
    while ind < len(data) and count_chunks < COUNT_BLOCK:
        chunks.append(' '.join(data[ind:ind + BLOCK_SIZE][:SEND_SIZE]))
        ind += BLOCK_SIZE
        count_chunks += 1

    mcqs = await asyncio.gather(*(get_mcq_each_block(thread.id, MCQ_GENERATOR_Assistant_id, chunk) for chunk in chunks))

    # deleting the thread
    response = client.beta.threads.delete(thread.id)
    print("[]. Deleting a thread")
    ic(response)

    return mcqs



# async def main():
#     data = """"""
#     data = data.split(" ")
#     mcqs = await generate(data)
#     # print("Done generaing")
#     # print(mcqs)
#     # for mcq in mcqs:
#     #     print(type(mcq))
#     #     ic(mcq)
#     #     print(type(mcq))
#     #     ic(mcq)

# mcqs = asyncio.run(main())
# for mcq in mcqs:
#     print(type(mcq))
    