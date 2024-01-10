import asyncio
from chronological import main, read_prompt, gather, fetch_max_search_doc, cleaned_completion, set_api_key
from dotenv import load_dotenv
import os
import openai
import time

# Load stuff
load_dotenv()
set_api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = os.getenv('OPENAI_API_KEY')


async def explanation(searchQuery):
    # Index 0
    p1 = f"Explain in informative terms to a non programmer in 300 words. {searchQuery}"
    explanationResult = await cleaned_completion(p1, engine="text-davinci-002", temperature=0.7, max_tokens=500,
                                                 top_p=1, frequency_penalty=0, presence_penalty=0)
    return explanationResult


async def roadmap(searchQuery):
    # Index 1
    p2 = f"Give a roadmap that is a series of instructions that someone should take to solve this question. {searchQuery} Include line breaks after each point. "
    roadmapResult = await cleaned_completion(p2, engine="text-davinci-002", temperature=0.7, max_tokens=500, top_p=1,
                                             frequency_penalty=0, presence_penalty=0)
    return roadmapResult


async def simpleLink(searchQuery, numResponses):
    # Index 2 +
    p3 = f"Return a link with a summary of well formatted code that solves this problem: {searchQuery}"
    simpleLinkResult = await gather(*[
        cleaned_completion(p3, engine="text-davinci-002", temperature=0.7, max_tokens=500, top_p=1,
                           frequency_penalty=0, presence_penalty=0) for n in range(numResponses)])
    return simpleLinkResult


async def resultsAsync(searchQuery):
    # simpleLink(searchQuery, numResults)
    tasks = [explanation(searchQuery), roadmap(searchQuery)]
    results = await asyncio.gather(*tasks)
    # resps = {'response': results[0], 'query': searchQuery, 'roadmap': results[1]}
    return results


from statistics import mean, median, mode, stdev, quantiles

asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
searchQuery = "How do you write a class in Java?"

logTime = []

print("------Chronology Test------")
for n in range(30):
    print(f"{n + 1}th Loop:")
    t2 = time.time()
    resps = asyncio.run(resultsAsync(searchQuery))
    t3 = time.time()
    print(t3 - t2)
    logTime.append(t3 - t2)
    print("\n")
print("------Results------")
quartiles = quantiles(logTime)
print("Mean: ", mean(logTime))
print("Q1: ", quartiles[0])
print("Median (Q2): ", quartiles[1])
print("Q3: ", quartiles[2])
print("Standard Deviation: ", stdev(logTime))