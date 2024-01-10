from langwave.document_loaders.sec import qk_html
import logging
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage
from langchain.chains import create_tagging_chain
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import get_openai_callback

from types import SimpleNamespace

log = logging.getLogger(__name__)


def get_num_tokens(text):
    llm = ChatOpenAI()
    num_tokens = llm.get_num_tokens_from_messages([HumanMessage(content=text)])
    return num_tokens


_TAGGING_TEMPLATE = """You are creating a rolling extract for the company. This will be used in a value proposition for investment.

Using logos, analytical rigor, calculations.

Extract the desired information from the following passage.

Maintain a concise and clear format, using shorthand and notations that only you, an LLM AI, can understand.

Include any magnitude units on monetary values. Keep monetary units in millions where applicable.

Only extract the properties mentioned in the 'information_extraction' function.

Passage:
{input}
"""


tagging_schema = {
    "properties": {
        "tags": {
            "type": "array",
            "description": "If you had to look up this passage later, what unique set of tags would you use?",
            "items": {"type": "string"},
        },
        "category": {
            "type": "string",
            "description": "What main financial category does this passage apply to?",
        },
        "subcategory": {
            "type": "string",
            "description": "Give a concise subcategory for the passage, if there is one, to distinguish this passage.",
        },
        "insight": {
            "type": "string",
            "description": "In a sentence, what is an insight of this passage?",
        },
        "analysis": {
            "type": "string",
            "description": "What would be the financial sentiment of this passage, from the perspective of an investor?",
            "enum": [
                "very positive",
                "positive",
                "mixed",
                "neutral",
                "negative",
                "very negative",
            ],
        },
        "title": {
            "type": "string",
            "description": "Give a well formatted title for this passage, so people know what they are looking at. Be concise.",
        },
    },
    "required": ["category", "subcategory", "tags", "insight", "analysis", "title"],
}


async def test_ner(args):
    s = SimpleNamespace()
    s.text = """*LIQUIDITY AND CAPITAL RESOURCES* *Liquidity and Financing Arrangements* Our principal sources of liquidity are from cash and cash equivalents, cash from operations, short-term borrowings under the credit agreement and our long-term financing. In November 2014, our Board of Directors authorized a $250 million stock repurchase program (the "2014 Program"). In November 2015, our Board of Directors approved the expansion of the 2014 Program by an additional $150 million. In August 2018, our Board of Directors approved the further expansion of the existing 2014 Program by an additional $150 million. In August 2022, our Board of Directors approved the further expansion of the existing 2014 Program by an additional $500 million. As of March 31, 2023, we had repurchased 8,712,998 shares of our common stock for an aggregate purchase price of approximately $573 million under the 2014 Program. During the three months ended March 31, 2023, we repurchased 201,742 shares of our common stock under the 2014 Program. Purchases under the 2014 Program may be made either through the open market or in privately negotiated transactions. Decisions regarding the amount and the timing of purchases under the 2014 Program will be influenced by our cash on hand, our cash flows from operations, general market conditions and other factors. The 2014 Program may be discontinued by our Board of Directors at any time. On October 4, 2018, Westlake Chemical Partners LP ("Westlake Partners") and Westlake Chemical Partners GP LLC, the general partner of Westlake Partners, entered into an Equity Distribution Agreement with UBS Securities LLC, Barclays Capital Inc., Citigroup Global Markets Inc., Deutsche Bank Securities Inc., RBC Capital Markets, LLC, Merrill Lynch, Pierce, Fenner & Smith Incorporated and Wells Fargo Securities, LLC to offer and sell WLK Partners common units, from time to time, up to an aggregate offering amount of $50 million. This Equity Distribution Agreement was amended on February 28, 2020 to reference a new shelf registration and subsequent renewals thereof for utilization under this agreement. No common units have been issued under this program as of March 31, 2023. We believe that our sources of liquidity as described above are adequate to fund our normal operations and ongoing capital expenditures and turnaround activities. Funding of any potential large expansions or potential acquisitions or the redemption of debt may likely necessitate, and therefore depend on, our ability to obtain additional financing in the future. We may not be able to access additional liquidity at favorable interest rates due to volatility of the commercial credit markets."""

    llm = ChatOpenAI(temperature=0.2, verbose=False, model="gpt-3.5-turbo-16k")

    chain = create_tagging_chain(
        tagging_schema, llm, prompt=ChatPromptTemplate.from_template(_TAGGING_TEMPLATE)
    )
    with get_openai_callback() as cb:
        x = await chain.arun(s.text)
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")

    log.info(f"test extracted: {x}")


async def test_ner_sections(args):
    llm = ChatOpenAI(temperature=0.2, verbose=False)

    chain = create_tagging_chain(
        tagging_schema, llm, prompt=ChatPromptTemplate.from_template(_TAGGING_TEMPLATE)
    )

    sections = qk_html.get_sections(args.filing)
    for s in sections:
        s.text = s.text.replace("\n", " ")
        num_tokens = llm.get_num_tokens_from_messages([HumanMessage(content=s.text)])
        print(f"### section {s.cnt}:\n`{s.text}`\nnum_tokens {num_tokens}")

        x = await chain.arun(s.text)

        log.info(f"extracted: {x}")


async def test_get_filing(args):
    llm = ChatOpenAI()
    sections = qk_html.get_sections(args.filing)

    # return

    for s in sections[:]:
        num_tokens = llm.get_num_tokens_from_messages([HumanMessage(content=s.text)])
        print(f"### section {s.cnt}:\n`{s.text}`\nnum_tokens {num_tokens}")

    log.info(f"have {len(sections)} sections")


async def main(args):
    # log.info(f"Reading {args.filing}")
    # await test_get_filing(args)
    # await test_ner_sections(args)
    await test_ner(args)
    # await test_get_filing(args)


import argparse, asyncio


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filing", "-f", help="filing document", required=False)
    parser.add_argument("--debug", "-d", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
