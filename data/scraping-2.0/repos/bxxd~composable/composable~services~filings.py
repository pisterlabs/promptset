import asyncio
import argparse
import yaml

from composable.cmn import utils
from composable.services._excerpts.pgvector import db
from langwave.document_loaders.sec import qk_html
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_tagging_chain
from langchain.schema.messages import HumanMessage
from langchain.callbacks import get_openai_callback
from composable.services.edgar import fetch_document
from datetime import datetime
from types import SimpleNamespace

import logging

log = logging.getLogger(__name__)

MODEL = "gpt-3.5-turbo-16k"


async def save_company_from_yaml(args):
    log.info(f"save_company: {args}")
    with open(args.company, "r") as file:
        data = yaml.safe_load(file)
        return await save_company(data)


async def save_company_from_cik_data(cik_data: dict):
    log.info(f"save_company: {cik_data}")
    cik = cik_data.get("cik_str")
    if not cik:
        log.error(f"save_company: cik not found")
        return
    cik = int(cik)

    data = {
        "cik": cik,
        "ticker": cik_data.get("ticker"),
        "name": cik_data.get("title"),
    }

    return await save_company(data)


async def save_company(data: dict):
    log.info(f"save_company: {data}")
    data = utils.remove_newlines_from_dict(data)

    cik = data.get("cik")

    async with db.Session.context() as session:
        company = await session.get_company_by_key(cik=cik)

        log.debug(f"get_company_by_key: {company}")
        if not company:
            company = db.Company.from_dict(data)
        else:
            company.update_from_dict(data)

        await session.save(company)
        log.debug(f"company: {company}")

        return company


async def save_filing_from_yaml(args):
    log.info(f"save_filing: {args}")

    with open(args.filing, "r") as file:
        data = yaml.safe_load(file)
        return save_filing(data)


async def save_filing(data: dict, model=MODEL):
    log.info(f"save_filing: {data}")
    cik = data.get("cik")
    ticker = data.get("ticker")
    # Convert 'filed_at' to datetime.date if it's a string
    filed_at = data.get("filed_at")
    if isinstance(filed_at, str):
        filed_at = datetime.strptime(filed_at, "%Y-%m-%d").date()
    # Convert 'reporting_for' to datetime.date if it's a string
    reporting_for = data.get("reporting_for")
    if isinstance(reporting_for, str):
        reporting_for = datetime.strptime(reporting_for, "%Y-%m-%d").date()
    filing_type = data.get("filing_type")
    if filing_type:
        filing_type = filing_type.lower()

    url = data.get("url")

    filing = None
    async with db.Session.context() as session:
        company = await session.get_company_by_key(cik=cik, ticker=ticker)
        if not company:
            log.critical(f"Company not found for cik: {cik} ticker: {ticker}")
            return
        filings = await session.get_filings_by_keys(
            company_id=company.id,
            url=url,
        )

        if not filings:
            log.info(f"Creating filing for cik: {cik} ticker: {ticker}")
            filing = db.Filing.from_dict(
                {
                    "filed_at": filed_at,
                    "filing_type": filing_type,
                    "reporting_for": reporting_for,
                    "url": url,
                    "company_id": company.id,
                    "model": model,
                }
            )
            filing.company_id = company.id
        else:
            (filing,) = filings[0]
            log.info(f"already have filing: {filing}")

        filing.model = MODEL

        filing = await session.merge(filing)
        await session.commit()

        return filing


async def save_excerpts(company, filing, model=MODEL):
    if not filing:
        log.error(f"save_filing: filing not found")
        return

    log.info(f"have filing: {filing}")

    async with db.Session.context() as session:
        await session.delete_excerpts(filing_id=filing.id)

        filing.status = "processing"
        filing = await session.merge(filing)
        await session.commit()

    log.info("here...")

    url = filing.url
    file = None
    if url:
        file = "/tmp/" + url.split("/")[-1]
        if utils.file_exists(file):
            log.info(f"save_filing: file already exists: {file}")
        else:
            content = await fetch_document(url)
            if content:
                log.info(f"save_filing: saving file: {file}")
                with open(file, "w") as f:
                    f.write(content)
    else:
        log.error(f"save_filing: file not found: {file} and no url provided")
        return

    try:
        cost = await save_filing_excerpts(company, filing, file, model)
    except Exception as e:
        log.error(f"save_filing: error: {e}")
        cost = 0.0
        async with db.Session.context() as session:
            filing.status = "error"
            filing = await session.merge(filing)
            await session.commit()
            return False

    async with db.Session.context() as session:
        filing.status = "processed"
        filing.cost = cost
        filing = await session.merge(filing)
        await session.commit()
        return True


_TAGGING_TEMPLATE = """You are creating a rolling extract for the company. This will be used in a value proposition for investment.

Using logos, analytical rigor, calculations.

Extract the desired information from the following passage.

Maintain a concise and clear format, using shorthand and notations that only you, an LLM AI, can understand.

Include any magnitude units on monetary values. Keep monetary units in millions where applicable.

Only extract the properties mentioned in the 'information_extraction' function.

Passage:
{input}
"""

# "tags": {
#             "type": "array",
#             "description": "If you had to look up this passage later, what unique set of tags would you use?",
#             "items": {"type": "string"},
#         },


tagging_schema = {
    "properties": {
        "category": {
            "type": "string",
            "description": "What is the main intent of this excerpt?",
        },
        "subcategory": {
            "type": "string",
            "description": "What is the main thing this excerpt is about?",
        },
        "insight": {
            "type": "string",
            "description": "What is the main thing people should know about this excerpt, if you could tell them one thing?",
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


async def process_section(
    section_text: str,
    llm: ChatOpenAI,
    chain,
    filing,
    company,
    session,
    section_count: int,
    report_title: str = None,
) -> (float, int):
    """Process a section and return the total cost"""
    log.info(f"process section {section_count}")
    section_text = section_text.replace("\n", " ")

    num_tokens = llm.get_num_tokens_from_messages([HumanMessage(content=section_text)])

    log.info(f"section: {num_tokens} tokens")

    # If token count exceeds 4096, split and recursively process
    if num_tokens > 4096:
        log.info(f"splitting section: {num_tokens}")
        halfway_point = len(section_text) // 2
        split_point = section_text.rfind(" ", 0, halfway_point)

        first_half = section_text[:split_point]
        second_half = section_text[split_point:]

        cost1, section_count = await process_section(
            first_half,
            llm,
            chain,
            filing,
            company,
            session,
            section_count,
            report_title,
        )
        cost2, section_count = await process_section(
            second_half,
            llm,
            chain,
            filing,
            company,
            session,
            section_count,
            report_title,
        )

        return cost1 + cost2, section_count

    # Current processing logic
    total_cost = 0.0

    section_count += 1

    with get_openai_callback() as cb:
        tags = await chain.arun(section_text)
        total_cost = cb.total_cost

    log.info(f"saving excerpt {section_count} for filing: {filing.id}")

    excerpt = await session.get_excerpt_by_keys(
        filing_id=filing.id, index=section_count
    )
    if not excerpt:
        excerpt = db.Excerpt(index=section_count, filing_id=filing.id)

    excerpt.excerpt = section_text
    excerpt.title = tags.get("title")
    excerpt.category = tags.get("category")
    excerpt.subcategory = tags.get("subcategory")
    excerpt.sentiment = tags.get("analysis")
    excerpt.insight = tags.get("insight")
    excerpt.cost = total_cost
    excerpt.tokens = num_tokens
    excerpt.company_name = company.name
    excerpt.company_ticker = company.ticker

    excerpt.filing_name = report_title

    excerpt = await session.merge(excerpt)
    await session.commit()

    log.info(f"excerpt: {excerpt} tags: {tags}")

    tags = tags.get("tags")
    if tags and isinstance(tags, list):
        tags = set(tags)
        await session.set_tags(excerpt.id, tags)

    await session.commit()

    return total_cost, section_count


import re


def split_closest_sentence(text):
    log.info(f"split_closest_sentence")
    halfway = len(text) // 2
    # Go halfway in terms of characters
    first_half = text[:halfway]
    second_half = text[halfway:]

    # Look for the nearest sentence end (., !, ?) to complete the first half
    match = re.search(r"[.!?]", second_half)
    if match:
        pos = match.start()
        first_half += second_half[: pos + 1]
        second_half = second_half[pos + 1 :]

    return first_half, second_half


def aggregate_sections(sections, llm):
    section_text = ""
    aggregated_sections = []
    max_tokens = 3072
    min_tokens = 1024

    log.info("aggregating sections...")

    for s in sections:
        projected_tokens = llm.get_num_tokens_from_messages(
            [HumanMessage(content=section_text + s.text)]
        )
        log.info(f"projected_tokens: {projected_tokens}")

        if projected_tokens > max_tokens:
            log.info("splitting section...")

            first_half, second_half = split_closest_sentence(section_text + s.text)
            new_s = SimpleNamespace()
            new_s.text = first_half
            new_s.tokens = llm.get_num_tokens_from_messages(
                [HumanMessage(content=new_s.text)]
            )
            new_s.cnt = len(aggregated_sections) + 1
            log.info("adding section... %d", new_s.cnt)
            aggregated_sections.append(new_s)
            section_text = second_half
        else:
            section_text += s.text
            if projected_tokens >= min_tokens:
                new_s = SimpleNamespace()
                new_s.text = section_text
                new_s.tokens = llm.get_num_tokens_from_messages(
                    [HumanMessage(content=new_s.text)]
                )
                new_s.cnt = len(aggregated_sections) + 1
                log.info("adding section... %d", new_s.cnt)
                aggregated_sections.append(new_s)
                section_text = ""

    if section_text:
        new_s = SimpleNamespace()
        new_s.text = section_text
        new_s.tokens = llm.get_num_tokens_from_messages(
            [HumanMessage(content=new_s.text)]
        )
        new_s.cnt = len(aggregated_sections) + 1
        log.info("adding remaining section... %d", new_s.cnt)
        aggregated_sections.append(new_s)

    return aggregated_sections


async def save_filing_excerpts(
    company: db.Company, filing: db.Filing, file: str, model: str = "gpt-3.5-turbo-16k"
):
    sections = qk_html.get_sections(file)

    llm = ChatOpenAI(temperature=0.2, verbose=False, model=model)

    sections = aggregate_sections(sections, llm)

    for s in sections:
        log.info(f"****** section: {s}")

    chain = create_tagging_chain(tagging_schema, llm)

    running_cost = 0.0
    current_section_count = 0

    upper_filing_period = filing.filing_period.upper()
    reporting_year = filing.reporting_for.year
    upper_filing_type = filing.filing_type.upper()

    report_title = f"{upper_filing_type} ({upper_filing_period} {reporting_year})"

    async with db.Session.context() as session:
        for s in sections:
            log.info(f"****** section: {s}")
            if not current_section_count:
                log.info(f"setting current_section_count {s.cnt}")
                current_section_count = s.cnt
            cost, current_section_count = await process_section(
                s.text,
                llm,
                chain,
                filing,
                company,
                session,
                current_section_count,
                report_title=report_title,
            )
            running_cost += cost
            # break

    return running_cost


async def main(args):
    log.debug(f"main: {args}")

    if args.debug:
        return

    if args.company:
        await save_company_from_yaml(args)

    if args.filing:
        await save_filing_from_yaml(args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filing", "-f", help="Filing config yaml")
    parser.add_argument("--company", "-c", help="Company config yaml")
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--model", "-m", default="gpt-3.5-turbo-16k")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
