import requests
from bs4 import BeautifulSoup
from rich.console import Console
from multiprocessing import Pool
import os
import openai
from dotenv import load_dotenv
import random
import time
from db import ScrapedData, session

class Scraper:
    def __init__(self, use_proxies=False):
        self.use_proxies = use_proxies
        self.completed_processes = 0
        self.total_execution_time = 0
        self.console = Console()
        load_dotenv()
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key is None:
            raise ValueError("OpenAI API key not found in environment variables.")
        openai.api_key = openai_key
        self.use_proxies = use_proxies
        if self.use_proxies:
            self.proxies = self.read_proxies_file('proxies.txt')

    def read_proxies_file(self, file_path):
        try:
            with open(file_path, 'r') as f:
                proxies = f.readlines()
        except Exception as e:
            self.console.print(f"[bold red]Failed to read proxies file. Error: {e}[/bold red]")
            return None
        return [proxy.strip() for proxy in proxies]

    def get_random_proxy(self):
        return random.choice(self.proxies)


    def scrape_html(self, url):
            try:
                if self.use_proxies:
                    proxy = self.get_random_proxy()
                    response = requests.get(url, timeout=30, proxies={'http': proxy, 'https': proxy})
                else:
                    response = requests.get(url, timeout=30)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                self.console.print(f"[bold red]Failed to retrieve the webpage. Error: {e}[/bold red]")
                return None

            try:
                soup = BeautifulSoup(response.content, 'html.parser')
            except Exception as e:
                self.console.print(f"[bold red]Failed to parse HTML content. Error: {e}[/bold red]")
                return None

            return str(soup)

    def analyze_html(self, html):
            try:
                initial_prompt = (f"Please analyze the following HTML content and extract the important information:\n\n{html}\n\n"
                          "Important information includes the abstract, introduction, development, conclusions, or any other relevant information.")
                initial_prompt_tokens = len(initial_prompt.split())
                if initial_prompt_tokens > 4096:
                    initial_prompt = (f"""Please analyze the following HTML content and extract the important information:\n\n{parts_list[0]}\n\n
                              Important information includes the abstract, introduction, development, conclusions, or any other relevant information, 
                              I will give you: {parts}, when you give me the important information from this query, I will continue asking you about the remaining parts, and at the end, I will ask you for all the relevant information you have found from the complete file, this is the first part.""")
                    parts = int(initial_prompt/ 4096) + 1
                    tokens_per_part = int(initial_prompt / parts)
                    parts_list = [html[i:i+tokens_per_part] for i in range(0, len(html), tokens_per_part)]
                    response = openai.Completion.create(
                        engine="davinci",
                        prompt=initial_prompt,
                        max_tokens=4096,
                        n=1,
                        stop=None,
                        temperature=0.5,
                    )
                    for i in range(1, parts):
                        prompt = (f"Please analyze the following HTML content and extract the important information:\n\n{parts_list[i]}\n\n"
                              "Important information includes the abstract, introduction, development, conclusions, or any other relevant information, this is the \n\n{part[i]}\n\n part.")
                        response = openai.Completion.create(
                            engine="davinci",
                            prompt=prompt,
                            max_tokens=4096,
                            n=1,
                            stop=None,
                            temperature=0.5,
                        )
                    prompt = (f"""Finally, please give me all the relevant information you have found from the complete file, this is the last part.""")
                    response = openai.Completion.create(
                        engine="davinci",
                        prompt=prompt,
                        max_tokens=4096,
                        n=1,
                        stop=None,
                        temperature=0.5,
                    )
                    important_info = response.choices[0].text.strip()
                    
                else:
                    prompt = (f"Please analyze the following HTML content and extract the important information:\n\n{html}\n\n"
                              "Important information includes the abstract, introduction, development, conclusions, or any other relevant information.")
                    response = openai.Completion.create(
                        engine="davinci",
                        prompt=prompt,
                        max_tokens=4096,
                        n=1,
                        stop=None,
                        temperature=0.5,
                    )
                    important_info = response.choices[0].text.strip()
                return important_info
            except Exception as e:
                self.console.print(f"[bold red]Failed to analyze HTML content. Error: {e}[/bold red]")
                return None
            

    def scrape_and_analyze(self, url):
            try:
                html = self.scrape_html(url)

                important_info = self.analyze_html(html)

                scraped_data = {
                    'url': url,
                    'html': html,
                    'important_info': important_info
                }

                return scraped_data
            except Exception as e:
                self.console.print(f"[bold red]Failed to scrape and analyze URL. Error: {e}[/bold red]")
                return None

    def read_links_file(self, file_path):
            try:
                with open(file_path, 'r') as f:
                    links = f.readlines()
            except Exception as e:
                self.console.print(f"[bold red]Failed to read links file. Error: {e}[/bold red]")
                return None
            return [link.strip() for link in links]

    def run(self):
            start_time = time.time()
            links = self.read_links_file('links.txt')
            if links:
                with Pool(os.cpu_count()) as p:
                    results = p.map(self.scrape_and_analyze, links)
                    self.completed_processes = len(results)
                    self.total_execution_time = time.time() - start_time
                    for result in results:
                        data = ScrapedData(
                            url=result['url'],
                            html=result['html'],
                            important_info=result['important_info'],
                            total_execution_time=self.total_execution_time,
                            completed_processes=self.completed_processes,
                            tokens_used=result['important_info'].split()
                        )
                        session.add(data)
                        session.commit()
            else:
                self.console.print("[bold red]No links found in links.txt file.[/bold red]")

if __name__ == '__main__':
        scraper = Scraper(use_proxies=True)
        scraper.run()

