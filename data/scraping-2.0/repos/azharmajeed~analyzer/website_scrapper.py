# import requests


# r = requests.get(
#     "https://www.livemint.com/market/live-blog/jio-financial-services-share-price-live-blog-for-30-aug-2023-11693363140475.html"
# )

# with open("some_text.txt", "w+") as f:
#     f.write(r.text)

import asyncio
from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.tools.playwright.utils import (
    create_async_playwright_browser,
    create_sync_playwright_browser,  # A synchronous browser is available, though it isn't compatible with jupyter.
)


async_browser = create_async_playwright_browser()
toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
tools = toolkit.get_tools()

tools_by_name = {tool.name: tool for tool in tools}
navigate_tool = tools_by_name["navigate_browser"]
get_elements_tool = tools_by_name["get_elements"]


# async def navigate_url(url):
#     await navigate_tool.arun({"url": url})
#     await get_elements_tool.arun(
#         {"selector": ".container__headline", "attributes": ["innerText"]}
#     )

url = "https://www.livemint.com/market/live-blog/jio-financial-services-share-price-live-blog-for-30-aug-2023-11693363140475.htmls"


async def async_load_playwright(url) -> str:
    from bs4 import BeautifulSoup
    from playwright.async_api import async_playwright

    results = ""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.goto(url)

            page_source = await page.content()
            soup = BeautifulSoup(page_source, "html.parser")

            for script in soup(["script", "style"]):
                script.extract()

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            results = "\n".join(chunk for chunk in chunks if chunk)
            print(results)
        except Exception as e:
            results = f"Error: {e}"
        await browser.close()
    return results


async def main():
    output = await async_load_playwright(url)
    print(output)


# Run the async function
asyncio.run(main())
