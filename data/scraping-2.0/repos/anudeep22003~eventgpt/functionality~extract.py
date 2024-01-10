import requests
import json
import re
from urllib.parse import urlparse, SplitResult, urlsplit
import openai
import time
from bs4 import BeautifulSoup
from collections import Counter
from usp.tree import sitemap_tree_for_homepage
from sqlalchemy.orm import Session
from functools import wraps
from functionality.sitemap import SitemapBuilder, is_url_media_type
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(f"logs/{__name__}.log", mode="a")
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)


#! to handle SSL: DH_KEY_TOO_SMALL] dh key too small (_ssl.c:1002) error
requests.packages.urllib3.util.ssl_.DEFAULT_CIPHERS += "HIGH:!DH:!aNULL"
try:
    requests.packages.urllib3.contrib.pyopenssl.DEFAULT_SSL_CIPHER_LIST += (
        "HIGH:!DH:!aNULL"
    )
except AttributeError:
    # no pyopenssl support used / needed / available
    pass

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}

from app import models, schemas, crud, db
from app.deps import get_db


def check_if_domain_exists(url: str) -> bool:
    "check if domain exists in the database"
    domain = urlparse(url).netloc
    with get_db() as db:
        domain_from_db = crud.domain.get_domain_by_name(db=db, domain=domain)
    if domain_from_db is None:
        return False
    return True


class IndexEventPage:
    def __init__(self, urlsplit_obj: SplitResult, freq_limit: int = 4) -> None:
        self.urlsplit_obj = urlsplit_obj
        self.start_time = time.perf_counter()
        self.sitemap = self.get_sitemap(urlsplit_obj)
        self.textrank = Counter()
        self.pagerank = Counter()
        self.page_index = {}
        self.freq_limit = freq_limit
        self.orchestrate()
        pass

    ######################################################
    ################ Utility Functions ################
    ######################################################

    def timer(func) -> None:
        "time the function"

        @wraps(func)
        def wrap_func(self, *args, **kwargs):
            start_time = time.time()
            results = func(self, *args, **kwargs)
            end_time = time.time()
            print(f"Time taken for {func.__name__}: {end_time - start_time} seconds")
            return results

        return wrap_func

    ######################################################

    def change_freq_limit(self, new_freq_limit: int) -> None:
        self.freq_limit = new_freq_limit

    def check_if_event_is_parsed(self) -> bool:
        "check is to load from db, or to perform a new parse"
        ...

    ##################################################
    ################ Domain Functions ################
    ###################################################

    @timer
    def get_sitemap(self, urlsplit_obj: SplitResult) -> list[str]:
        s = SitemapBuilder()
        sitemap, _ = s(urlsplit_obj=urlsplit_obj)
        return sitemap

    ###################################################
    ################ SiteUrl Functions ################
    ###################################################

    def orchestrate(self) -> None:
        "orchestrates the entire process of indexing"
        self.download_html_build_ranks()
        self.db_create_domain_entry()
        self.extract_clean_text()

    @timer
    def download_html_build_ranks(self) -> None:
        for pathurl in self.sitemap:
            # if already exists in db, then get from db
            with get_db() as db:
                siteurl = crud.siteurl.get_by_url(db=db, url=pathurl)
                if siteurl is None:
                    html_text = self.get_html_text(pathurl)
                elif not siteurl.html:
                    html_text = self.get_html_text(pathurl)
                else:
                    html_text = siteurl.html
                soup = self.build_soup(html_text)
                self.build_ranks(soup)
                # save html content to db?
                # make siteurl db entry for each url
                if siteurl is None:
                    self.db_create_siteurl_entry(url=pathurl, html=html_text)

    @timer
    def extract_clean_text(self) -> str:
        "extract for every url ranked text and save it to db"
        for url in self.sitemap:
            self.parse_html_store_db(url)

    def db_create_siteurl_entry(self, url: str, html: str = "", text: str = "") -> None:
        "create a siteurl entry in the database"
        #! assumes url_home is the homepage, user may pass a random page
        #! saving the entire url including path
        siteurl = schemas.SiteUrlCreate(
            url=url, domain=self.urlsplit_obj.netloc, text=text, html=html
        )
        with get_db() as db:
            # db = get_db()
            siteurl_from_db = crud.siteurl.create(db=db, obj_in=siteurl)
        return siteurl_from_db

    def get_html_text(self, urlpath: str) -> str:
        "download the html of the page"
        if urlpath == "/":
            urlpath = "https://" + self.urlsplit_obj
        urlsplit_obj = urlsplit(urlpath)
        if is_url_media_type(urlsplit_obj):
            return """
                <!DOCTYPE html>
                <html lang="en">
                <body>
                    None
                </body>
                </html>
            """
        #! highly unsafe, security risk, not verifying the ssl certificate with verify flag
        try:
            res = requests.get(urlpath, headers=headers)
        except requests.exceptions.SSLError:
            res = requests.get(urlpath, verify=False, headers=headers)
        except requests.exceptions.InvalidSchema:
            logger.info(f"Invalid schema for {urlpath}")
            return "None"
        except requests.exceptions.MissingSchema:
            urlpath = "https://" + self.urlsplit_obj.netloc + urlpath
            try:
                res = requests.get(urlpath, verify=False, headers=headers)
            except requests.exceptions.ConnectionError:
                return """
                    <!DOCTYPE html>
                    <html lang="en">
                    <body>
                        None
                    </body>
                    </html>
                """
        if res.status_code == 200:
            return res.text
        else:
            return """
                <!DOCTYPE html>
                <html lang="en">
                <body>
                    None
                </body>
                </html>
            """

    def build_soup(self, html_text: str) -> BeautifulSoup:
        "build a soup object from the html text"
        return BeautifulSoup(html_text, "html.parser")

    def build_ranks(self, soup: BeautifulSoup) -> None:
        "add to both textrank and pagerank"
        # build pagrank
        for link in soup.find_all("a"):
            self.pagerank.update([link.get("href")])

        # build textrank
        for string in soup.stripped_strings:
            newline_removed = re.sub("\n+", " ", string)
            whitespace_removed = re.sub("\s+", " ", newline_removed)
            self.textrank.update([str(whitespace_removed)])

    def ranked_parse(self, soup: BeautifulSoup) -> str:
        "parse the html using textrank"
        ranked = []
        limit = self.freq_limit
        for string in soup.stripped_strings:
            if self.textrank[string] <= limit:
                if soup.find(re.compile("^h"), string=string):
                    ranked.append(f"\n\n{string}:\n")
                elif soup.find(re.compile("^li"), string=string):
                    ranked.append(f"- {string}")
                else:
                    ranked.append(string)

        return "\n".join(ranked)

    def parse_html_store_db(self, url: str) -> str:
        """
        Use textrank frequency mapping to parse high informational content
        Load the html from sqlite, rather than downloading again if it exists
        #! not performing any checks if the domain being investigated is indexed or not
        #! do you even need to perform above check?
        """
        with get_db() as db:
            # db = get_db()
            siteurl = crud.siteurl.get_by_url(db=db, url=url)
        #! i think the below check is unnecessary
        if siteurl is None:
            html_text = self.get_html_text(url)
            siteurl = self.db_create_siteurl_entry(url=url, html=html_text)
        else:
            html_text = siteurl.html
        soup = self.build_soup(html_text)
        text = self.ranked_parse(soup)
        with get_db() as db:
            siteurl = crud.siteurl.update_text(db=db, url=url, text=text)

    def db_get_text(self, url: str) -> str:
        #! handle not finding the entry and getting None
        with get_db() as db:
            # db = get_db()
            siteurl = crud.siteurl.get_by_url(db=db, url=url)
        return siteurl.text

    ######################################################
    ################ EndofIndex Functions ################
    ######################################################

    def db_create_domain_entry(self) -> models.Domain:
        "create a domain entry in the database"
        pagerank = self.pagerank
        textrank = self.textrank
        sitemap = " , ".join(self.sitemap)

        domain_in = schemas.DomainCreate(
            domain=self.urlsplit_obj.netloc,
            pagerank=pagerank,
            textrank=textrank,
            sitemap=sitemap,
            time_to_index=round(time.perf_counter() - self.start_time, 2),
        )
        with get_db() as db:
            #! to be fully right you should maybe update the textran and pagelist if anychanges are detected
            # if domain entry already exists, do nothing else create
            domain_from_db = crud.domain.get_domain_by_name(
                db=db, domain=self.urlsplit_obj.netloc
            )
            if domain_from_db is None:
                domain_from_db = crud.domain.create(db=db, obj_in=domain_in)
        return domain_from_db


if __name__ == "__main__":
    home_url = input("Enter the home url: ")
    start_time = time.time()
    i = IndexEventPage(url_home=home_url)
    end_time = time.time()
    print(f"Time taken to build index: {end_time - start_time} seconds")
    print(f"Number of urls in sitemap: {len(i.sitemap)}")
    while True:
        print("-" * 50)
        site_url_text = input("Enter the url to parse: ")
        print("-" * 50)
        print("-" * 50, end="\n\n")
