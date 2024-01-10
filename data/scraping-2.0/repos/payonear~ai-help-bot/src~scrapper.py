import logging
import time
from datetime import datetime, timedelta

import bs4
from selenium import webdriver
from telegram.utils.helpers import escape_markdown
from webdriver_manager.chrome import ChromeDriverManager

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger("tcpserver")
driver = webdriver.Chrome(ChromeDriverManager().install())
since_when = datetime.now() - timedelta(days=2)


class Scraper:
    def __init__(self) -> None:
        logger.info("Scraper is defined!")

    def __str__(self) -> str:
        print(f"Scraper object for Facebook AI, Google AI and OpenAI blogs.")

    def __scrape_facebook(self) -> list:
        logger.info("Starting scraping Facebook...")
        fb_todays_posts = []
        url = "https://ai.facebook.com/blog/"
        driver.get(url)
        try:
            time.sleep(10)
            soup = bs4.BeautifulSoup(driver.page_source, "html.parser")
            top_post = soup.find_all("div", {"class": "_8x7i _8x8q _8x92"})
            other_posts = soup.find_all("div", {"class": "_8wpt"})
            all_posts = top_post + other_posts
            for post in all_posts:
                elements = [s for s in post.strings]
                title, descr, date = elements[-3:]
                date = datetime.strptime(date, "%B %d, %Y")
                domain = "|".join(elements[:-3])

                if date > since_when:
                    date = date.strftime("%B %d, %Y")
                    logger.info("Recent post found from Facebook!")
                    link = post.find("a", href=True)["href"]
                    org = "Facebook AI"
                    p = (org, domain, date, title, descr)
                    (org, domain, date, title, descr) = [escape_markdown(el, version=2) for el in p]
                    message = f"{domain}\n{org} \\- {date}\n\n*{title}*\n_{descr}_"
                    fb_todays_posts.append((message, link))
                else:
                    logger.info("No other posts found for Facebook.")
                    break
        except:
            logger.info(f"{url} is not responding on time.")

        return fb_todays_posts

    def __scrape_google(self) -> list:
        logger.info("Starting scraping Google...")
        google_todays_posts = []
        url = "https://ai.googleblog.com/"
        driver.get(url)
        try:
            time.sleep(10)
            soup = bs4.BeautifulSoup(driver.page_source, "html.parser")
            posts = soup.find_all("div", {"class": "post"})
            for post in posts:
                title = post.find("a")["title"]
                date = post.find("span", {"class": "publishdate"}).text
                date = date.strip().split(",")[1:]
                date = ",".join(date).strip()
                date = datetime.strptime(date, "%B %d, %Y")

                if date > since_when:
                    date = date.strftime("%B %d, %Y")
                    logger.info("Recent post found from Google!")
                    link = post.find("a", href=True)["href"]
                    org = "Google AI"
                    p = (org, date, title)
                    (org, date, title) = [escape_markdown(el, version=2) for el in p]
                    message = f"{org} \\- {date}\n\n*{title}*"
                    google_todays_posts.append((message, link))
                else:
                    logger.info("No other posts found for Google.")
                    break
        except:
            logger.info(f"{url} is not responding on time.")

        return google_todays_posts

    def __scrape_openai(self) -> list:
        logger.info("Starting scraping OpenAI...")
        openai_todays_posts = []
        url = "https://openai.com/blog/"
        driver.get(url)
        try:
            time.sleep(10)
            soup = bs4.BeautifulSoup(driver.page_source, "html.parser")
            posts = soup.find_all("div", {"class": "post-card-full medium-xsmall-copy"})
            for post in posts:
                title = post.find("a", href=True).text
                date = post.find("time").text
                date = datetime.strptime(date, "%B %d, %Y")

                if date > since_when:
                    date = date.strftime("%B %d, %Y")
                    logger.info("Recent post found from OpenAI!")
                    link = post.find("a", href=True)["href"].replace("/blog", "")
                    link = url + link
                    descr = self.__scrape_openai_post_descr(link)
                    org = "OpenAI"
                    p = (org, date, title, descr)
                    (org, date, title, descr) = [escape_markdown(el, version=2) for el in p]
                    message = f"{org} \\- {date}\n\n*{title}*\n_{descr}_"
                    openai_todays_posts.append((message, link))
                else:
                    logger.info("No other posts found for OpenAI.")
                    break
        except:
            logger.info(f"{url} is not responding on time.")

        return openai_todays_posts

    def __scrape_openai_post_descr(self, link):
        logger.info("Scraping description for OpenAI post!")
        driver.get(link)
        try:
            time.sleep(10)
            soup = bs4.BeautifulSoup(driver.page_source, "html.parser")
            descr = soup.find("meta", property="og:description")
            descr = descr.get("content")
        except:
            logger.info(f"{link} is not responding on time.")

        return descr

    def scrape_all(self) -> list:
        all_todays_posts = []
        all_todays_posts.extend(self.__scrape_facebook())
        all_todays_posts.extend(self.__scrape_google())
        all_todays_posts.extend(self.__scrape_openai())
        return all_todays_posts
