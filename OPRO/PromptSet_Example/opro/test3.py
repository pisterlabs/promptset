import asyncio
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from playwright.async_api import async_playwright
from playwright._impl._errors import Error as PlaywrightError
from googlesearch import search
from tqdm import tqdm
import time

# urls = [
#     "https://scrapethissite.com/pages/ajax-javascript/#2015",
#     "https://scrapethissite.com/pages/ajax-javascript/#2014",
#     "https://scrapethissite.com/pages/ajax-javascript/#2013",
# ]

async def open_page(context, url):
    try:
        page = await context.new_page()
        await page.goto(url) 
        await page.wait_for_load_state(state="networkidle")
        await page.wait_for_timeout(1000)
    except PlaywrightError as e:
        print(f"Failed to load page due to Playwright error: {e}. Skipping Page...")
    except TimeoutError:
        print("Failed to load page. Skipping Page...")
    # finally:
    #     await page.close()
    
    
async def run(urls):
    # Use playwright async API
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        # Set default timeout to 10 seconds
        context.set_default_timeout(1000)
        
        await asyncio.gather(*(open_page(context, url) for url in urls))
        all_text = ""            
        
        for _ in tqdm(range(len(context.pages))):            
            # Get the page content
            html_content = await context.pages[0].content()
            print(context.pages[0].url)
            
            # Assuming you have BeautifulSoup installed
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract all text
            text = soup.get_text()
            all_text += "\n" + text
            await context.pages[0].close()         

        await browser.close()
    return all_text
        
if __name__ == "__main__":
    start = time.time()
    res = list(search("Color of the apple fruit", advanced=True, num_results=20))
    urls = [r.url for r in res]
    out = asyncio.run(run(urls))
    print(len(out))
    finish = time.time()
    print(f"Time taken: {finish - start} seconds")
