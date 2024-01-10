# import polygon
import datetime
import pprint
from dateutil.relativedelta import relativedelta
import sys
from polygon import Polygon

import openai
sys.path.insert(0, r"C:\Users\isaac\OneDrive\Desktop\Todos\Codes\Real world projects\Stock AI")
import AI
import asyncio



news = ["Warren buffet dies", "U.S. makes treaty to resume russian oil imports to the U.S.", "GPT-5 was released"]

async def main():


    await AI.GPT("how many seconds in an hour")


# asyncio.run(main())

p = Polygon()

print(p.getPrices("AMZN", datetime.datetime(2023, 2, 2)))










