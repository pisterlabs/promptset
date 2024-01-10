from dotenv import load_dotenv
from langchain.utilities import GoogleSerperAPIWrapper
from generate_structured_data import GenerateStructuredData as GSD
import asyncio
import aiohttp
from unstructured.partition.html import partition_html
from generate_document import GenerateDocument as gd
import time

load_dotenv()

google = GoogleSerperAPIWrapper()


urls = [
    "https://www.newser.com/story/341661/china-preps-for-assault-with-tips-learned-from-russia.html",
    "https://www.newser.com/story/341701/aid-reaches-gaza-as-us-issues-a-warning.html",
    "https://www.reddit.com/r/Python/comments/17dkshe/when_have_you_reach_a_python_limit/",
    "https://www.msn.com/en-ca/news/canada/india-says-relations-with-canada-passing-through-difficult-phase/ar-AA1iEsvJ?ocid=winp2fptaskbar&cvid=4f3be7e3697a4beba7f62d1f931de72a&ei=4",
]


tasks = []

from multiprocessing.dummy import Pool as ThreadPool


def get_doc(url):
    gsd = GSD()
    content = gd.generate(url)
    formatted_input = gsd.generate_input(content)
    return formatted_input


def main():
    start = time.time()
    # for i in urls:
    #     tasks.append(asyncio.create_task(get_doc(i)))

    # r = await asyncio.gather(*tasks)

    # print(r)

    pool = ThreadPool(20)

    # open the urls in their own threads
    # and return the results
    r = pool.map(get_doc, urls)

    print(r)

    # close the pool and wait for the work to finish
    pool.close()
    pool.join()

    end = time.time()
    print("Runtime: " + str(end - start))


main()

# loop = asyncio.get_event_loop()
# loop.run_until_complete(main())
