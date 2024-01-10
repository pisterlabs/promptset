import argparse
import os
import asyncio
from dotenv import load_dotenv

from module_openai.openai_async_operator import AsyncOpenAIOperator
from module_openai.openai_operator import OpenAIOperator
from prompts.generic_prompts import MEDICINE_INSTRUCTIONS
from prompts.utils import generate_prompts
from prompts.my_research_prompts import research_microbiome


async def main(prompts: list, folder_name: str, file_name: str):
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    operator = AsyncOpenAIOperator(api_key)

    await operator.create_assistant(
        name="Med Book Builder",
        instructions=MEDICINE_INSTRUCTIONS
    )

    await operator.start_thread()
    await operator.create_messages(prompts)
    runs = await operator.execute_run()
    for run in runs:
        operator.show_json(run)
    await operator.list_messages(folder_name, file_name)


def run_synchronous(prompt: str):
    # Example usage:
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    operator = OpenAIOperator(api_key)

    operator.create_assistant(
        name="Data visualizer",
        instructions=MEDICINE_INSTRUCTIONS
    )

    operator.start_thread()
    operator.create_message(prompt)
    run = operator.execute_run()
    operator.show_json(run)
    return operator.list_messages()


parser = argparse.ArgumentParser(description="Example Argparse Program")
parser.add_argument("--run_sync", action="store_true", help="...")

args = parser.parse_args()

do_run_sync = args.run_sync
subject = 'Gut Microbiome'

print(f"run_sync: {args.run_sync}")

prompts = generate_prompts(research_microbiome, subject=subject)

# import ipdb; ipdb.set_trace()
if not do_run_sync:
    asyncio.run(main(prompts[:3], folder_name="med_book", file_name=subject))

elif do_run_sync:
    for prompt in prompts:
        run_synchronous(prompt)
