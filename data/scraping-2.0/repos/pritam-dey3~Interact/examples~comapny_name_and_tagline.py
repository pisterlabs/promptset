import asyncio
import os

import openai
from dotenv import load_dotenv

from interact.base import Cascade, Handler, Message
from interact.handlers import OpenAiLLM

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]


class CompanyNamePrompt(Handler):
    role = "CompanyNameGenerator"
    prompt = (
        "What would be an appropriate name for a business specializing in {product}?"
    )

    async def process(self, msg: Message, csd: Cascade) -> str:
        new_msg = self.prompt.format(product=msg.primary)
        csd.vars["product"] = msg.primary
        return new_msg


class CompanyTaglinePrompt(Handler):
    role = "CompanyTaglineGenerator"
    prompt = (
        "What would be an appropriate tagline for a business specializing in {product}"
        " and with company name {company_name}?\nFormat your output in the following"
        " format:\n<company_name>: <tagline>"
    )

    async def process(self, msg: Message, csd: Cascade) -> str:
        new_msg = self.prompt.format(
            company_name=msg.primary, product=csd.vars["product"]
        )
        return new_msg


name_and_tagline_generator = (
    CompanyNamePrompt() >> OpenAiLLM() >> CompanyTaglinePrompt() >> OpenAiLLM()
)

print(asyncio.run(name_and_tagline_generator.start("socks")).last_msg)
# >> The Sock Spot: Step into Comfort
print(asyncio.run(name_and_tagline_generator.start("socks")).last_msg)
# >> The Sock Spot: Step into Comfort
