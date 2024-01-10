import random
import string
import logging

import openai
from scrapy.exceptions import DropItem
from sqlalchemy.dialects.postgresql import insert as postgres_upsert
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert

from api.database import Base, SessionLocal, engine
from api.models import AlphaStreet, YouTube, MoneyControl
from crawler.exceptions import NoDescriptionMoneycontrolException
from config import config


CHATGPT_PROMPT = (
    "Create a summary in no more than two sentences,"
    "remove all references to numbers and figures,"
    "instead highlight the most important point in simple words"
)


class DuplicatesPipeline:
    def __init__(self):
        self.db = SessionLocal()
        self.seen = set()

        logging.info("Loading links from the database")

        # fetch all the links from the database, from all the tables
        for table in (AlphaStreet, YouTube, MoneyControl):
            for item in self.db.query(table.link).all():
                self.seen.add(item[0])

        logging.info(f"Loaded {len(self.seen)} links from the database")
        logging.info(f"Size in memory in megabyte: {self.seen.__sizeof__() / (1024 ** 2)}")

    def process_item(self, item, spider):
        link = item["link"]
        if link in self.seen:
            raise DropItem(f"Duplicate item found: {item}")
        else:
            self.seen.add(link)
            return item


class ShortLinkPipeline:
    def __init__(self):
        self.db = SessionLocal()

        # to find duplicates
        self.short_codes = set()

        # charset and length for generating short links
        self.charset = string.ascii_letters + string.digits
        self.short_code_length = 6

        logging.info("Loading links from the database")

        # fetch all the short_codes from the database, from all the tables
        for table in (AlphaStreet, YouTube, MoneyControl):
            for item in self.db.query(table.short_code).all():
                self.short_codes.add(item[0])

        logging.info(f"Loaded {len(self.short_codes)} short codes from the database")
        logging.info(f"Size in memory in megabyte: {self.short_codes.__sizeof__() / (1024 ** 2)}")

    def process_item(self, item, spider):
        item["short_code"] = self.generate_short_code()
        return item

    def generate_short_code(self):
        # Generate a short link by taking random strings of length 6
        short_code = "".join(random.sample(self.charset, k=self.short_code_length))

        # If the short link is already in the database, generate a new one recursively
        if short_code in self.short_codes:
            return self.generate_short_code()

        # Add the short link to the set of short links and return it
        self.short_codes.add(short_code)
        return short_code


class SummaryPipeline:
    @staticmethod
    async def summarize(text, prompt=CHATGPT_PROMPT, tries=0):
        try:
            openai.api_key = config.OPENAI_API_KEY
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text},
                ],
                temperature=1,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
        except openai.error.OpenAIError as e:
            # if the api call fails, try again, but only 3 times
            if tries >= 3:
                raise e
            return await SummaryPipeline.summarize(text, prompt, tries + 1)

        # return the summary
        return response.choices[0].message.content

    async def process_item(self, item, spider):
        # Count the number of chatgpt api calls, to the scrapy stats
        # not working for some reason
        # spider.crawler.stats.inc_value("chatgpt_api_calls")

        # only generate summary for articles if they have content
        if content := item.get("content"):
            try:
                summary = await self.summarize(content)
            except Exception as e:
                logging.error(f"Failed to summarize article: {item['link']}")
                logging.error(e)
                summary = None
            else:
                item["summary"] = summary
        return item


class DatabasePipeline:
    def __init__(self):
        self.db = SessionLocal()
        self.items = []

        # Create tables if they don't exist
        Base.metadata.create_all(bind=engine)

    def process_item(self, item, spider):
        if not item.get("date"):  # because some alphastreet items don't have date
            raise ValueError("Item has no date", item)

        if spider.name == "moneycontrol":
            if not item.get("description"):
                raise NoDescriptionMoneycontrolException(item)
            if not item.get("content"):
                raise ValueError("Moneycontrol Item has no content", item)

        self.items.append(dict(item))
        return item

    def close_spider(self, spider):
        logging.info(f"Inserting {len(self.items)} items into the database")
        logging.info(f"Size in memory in megabyte: {self.items.__sizeof__() / (1024 ** 2)}")
        if not self.items:
            raise ValueError("No items to insert")

        # TODO: WTF IS THIS; PLS CLEANUP THIS MESS
        # item type of the items in the list
        item_type = {
            "alphastreet": AlphaStreet,
            "youtube": YouTube,
            "moneycontrol": MoneyControl,
        }[spider.name]

        # create an upsert statement
        # https://docs.sqlalchemy.org/en/20/orm/queryguide/dml.html#orm-upsert-statements
        if engine.name == "sqlite":
            stmt = sqlite_upsert(item_type).values(self.items)
        elif engine.name == "postgresql":
            stmt = postgres_upsert(item_type).values(self.items)
        else:
            raise NotImplementedError(f"Engine {engine.name} not supported")

        # get the columns of the table to be updated, except link and scraped_date
        columns = {}
        for c in item_type.__table__.columns:
            if c.name not in ("link", "scraped_date", "short_code"):
                columns[c.name] = getattr(stmt.excluded, c.name)

        # TODO: FIND SOMETHING BETTER; THIS WILL FAIL IF LINK IS NOT THE PRIMARY KEY
        # on conflict do update using link as the unique constraint
        stmt = stmt.on_conflict_do_update(index_elements=["link"], set_=columns)
        print(stmt.compile(compile_kwargs={"literal_binds": True}))
        self.db.execute(stmt)

        # commit and close the database
        self.db.commit()
        self.db.close()
