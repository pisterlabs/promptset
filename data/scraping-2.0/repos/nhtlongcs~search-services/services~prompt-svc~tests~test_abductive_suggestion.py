import logging
import requests
import openai 
import os 
from dotenv import load_dotenv; load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
import asyncio
from features.abductive import AbductiveChain

def pprint(result):
    for i, r in enumerate(result):
        print(f"{i+1}. {r}")

async def test_abductive_chain():
    pprint(await AbductiveChain().execute_wo_rate_limit("I was praying to small golden Buddha in a tunnel."))
    print('='*80)
    pprint(await AbductiveChain().execute_wo_rate_limit("Boarding pass for PVG."))
    print('='*80)
    pprint(await AbductiveChain().execute_wo_rate_limit("breakfast, Christmas Day"))
    print('='*80)

asyncio.run(test_abductive_chain())