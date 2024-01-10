from datetime import date, datetime, timedelta
from typing import Dict, List

import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import AsyncHtmlLoader
from tqdm import tqdm

from dags.modules.collect.base import BaseCollector
from dags.modules.database.pymongo import PymongoClient

SWCON_TARGET_LINK = "http://swcon.khu.ac.kr/wordpress/post/?mode=list&board_page="


class SwconCollector(BaseCollector):
    today_date = date.today()
    tomarrow_date = date.today() + timedelta(days=1)

    def __init__(self, target_link: str = SWCON_TARGET_LINK):
        self.links = []
        self.documents = []
        self.target_link = target_link

    def collect(
        self,
        start_date: datetime.date = today_date,
        end_date: datetime.date = tomarrow_date,
        max_page: int = 35,
    ) -> List[Dict]:
        # TODO: Serve progress informations.
        self.links = self._get_links(start_date, end_date, max_page=max_page)
        self.documents = self._get_documents(self.links)
        return self.documents

    def upload_db(self, db_host: str = "localhost", db_port: int = 27017) -> None:
        client = PymongoClient(host=db_host, port=db_port)
        client.insert_documents(self.documents)

    def _get_links(
        self, start_date: datetime.date, end_date: datetime.date, max_page: int = 35
    ):
        # TODO: How to determine max page?
        links = []
        is_break = False
        for page in tqdm(range(1, max_page)):
            if is_break:
                break
            response = requests.get(f"{self.target_link}{page}")
            soup = BeautifulSoup(response.text, "html.parser")
            for item in soup.select("#DoubleMajor_board_body > tr"):
                link = item.select_one("td.text-left > a").get("href")

                date_text = item.select("td:nth-child(4) > span")[0].getText()

                # If the item contains a colon, like 9:45, it means the datetime is today.
                if ":" in date_text:
                    content_date = datetime.today().date()
                else:
                    content_date = datetime.strptime(date_text, "%Y-%m-%d").date()

                # If the item is a notice, it must continue to do a full scan without stopping.
                if item.select(".mb-notice"):
                    if start_date <= content_date <= end_date:
                        links.append(link)
                else:
                    if end_date < content_date:
                        continue
                    elif start_date > content_date:
                        is_break = True
                        break
                    else:
                        links.append(link)
        return links

    def _get_documents(self, links):
        loader = AsyncHtmlLoader(links)
        documents = loader.load()
        transform_documents = self._transform_documents(
            documents, scope_selector=".content-box"
        )
        json_documents = self._convert_to_json(transform_documents)
        return json_documents
