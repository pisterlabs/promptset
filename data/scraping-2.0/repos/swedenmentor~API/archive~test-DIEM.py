import time
import requests
from bs4 import BeautifulSoup
import datetime
import jsonlines
from urllib.parse import urljoin
from googletrans import Translator
from langdetect import detect
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Define function to chunk text
def chunk_text(input_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_text(input_text)

class Crawler:
    def __init__(self):
        self.visited_urls = set()
        self.translator = Translator()
        self.data_buffer = []

    def write_to_file(self, output_file):
        if self.data_buffer:
            with jsonlines.open(output_file, mode='a') as file:
                for entry in self.data_buffer:
                    file.write(entry)
                self.data_buffer = []

    def translate_text(self, text):
        output = []
        for chunk in chunk_text(text):
            detected_lang = detect(chunk)  # Detect the language of the chunk
            if detected_lang in ['en', 'sv', 'vi']:  # Check if the detected language is one of the three
                if detected_lang != 'en':
                    translated_chunk = self.translator.translate(chunk, src=detected_lang, dest='en').text
                    output.append(translated_chunk)
                else:
                    output.append(chunk)
        return output

    def crawl_website(self, url, output_file, depth=5):
        if (depth == 0) or (url in self.visited_urls) or not url.startswith('https://www.migrationsverket.se'):
            return

        try:
            response = requests.get(url)
            if not response.status_code == 200:
                print(f'Non success status for url {url}')
                return
            self.visited_urls.add(url)

            soup = BeautifulSoup(response.text, 'html.parser')
            date_tag = soup.find('p', class_='ahjalpfunktioner')
            if date_tag:
                time_tag = date_tag.find('time')
                date = time_tag.get_text() if time_tag else datetime.datetime.now().strftime("%Y-%m-%d")
            else:
                date = datetime.datetime.now().strftime("%Y-%m-%d")

            title = soup.title.text.strip()
            title = self.translate_text(title)
            paragraphs = soup.find_all(['p', 'h1', 'h2'])

            text = "\n".join([p.get_text().strip() for p in paragraphs])
            chunks = self.translate_text(text)

            entries = {}
            for idx, chunk in enumerate(chunks):
                entries[str(idx)] = {
                    "chunk-id": str(idx),
                    "source": url,
                    "title": title,
                    "chunk": chunk,
                    "updated": date,
                }
            for key in entries:
                self.data_buffer.append(entries[key])
            self.write_to_file(output_file)

            links = soup.find_all('a')
            for link in links:
                href = link.get('href')
                if href and href.startswith('http'):
                    new_url = href
                else:
                    new_url = urljoin(url, href)
                if new_url not in self.visited_urls:
                    time.sleep(1)
                    self.crawl_website(new_url, output_file, depth=depth - 1)

        except requests.exceptions.RequestException as err:
            print(f"RequestException: {err}")
        except requests.exceptions.HTTPError as errh:
            print(f"HTTPError: {errh}")
        except requests.exceptions.ConnectionError as errc:
            print(f"ConnectionError: {errc}")
        except Exception as e:
            print(f"An error occurred while processing {url}: {str(e)}")

if __name__ == '__main__':
    crawler = Crawler()
    max_depth = 5
    url = 'https://www.migrationsverket.se/'
    output_file = 'migrationsverket.jsonl'

    # Crawling
    crawler.crawl_website(url, output_file=output_file, depth=max_depth)
