from datetime import date, datetime, timedelta
from typing import Dict, List

import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import AsyncHtmlLoader
from tqdm import tqdm

from dags.modules.collect.base import BaseCollector
from dags.modules.database.pymongo import PymongoClient

SWEDU_TARGET_LINK = "https://swedu.khu.ac.kr/bbs/board.php?bo_table=07_01&page="


class SweduCollector(BaseCollector):
    today_date = date.today()
    tomarrow_date = date.today() + timedelta(days=1)

    def __init__(self, target_link: str = SWEDU_TARGET_LINK):
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

    def upload_db(self, db_host="localhost", db_port=27017) -> None:
        client = PymongoClient(host=db_host, port=db_port)
        client.insert_documents(self.documents)

    def _get_links(self, start_date, end_date, max_page: int = 34):
        # TODO: How to determine max page?
        links = []
        is_break = False
        for page in tqdm(range(1, max_page)):
            if is_break:
                break
            response = requests.get(f"{self.target_link}{page}")
            soup = BeautifulSoup(response.text, "html.parser")
            for item in soup.select("#fboardlist > div > table > tbody > tr"):
                link = item.find("a").get("href")
                content_date = datetime.strptime(
                    item.select(".td_datetime")[0].getText().strip(), "%Y-%m-%d"
                ).date()

                # If the item is a notice, it must continue to do a full scan without stopping.
                if item.select(".notice_icon"):
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
            documents, scope_selector="#bo_v_atc", ignore_images=True, get_image=False
        )
        json_documents = self._convert_to_json(transform_documents)
        return json_documents
