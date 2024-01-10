import openai
import facebook_scraper

from app.config import Config
from app.domain.chatgpt import ChatGPT
from app.domain.facebook_groups_scraper import FacebookGroupsScraper
from app.services.advertisment_service import AdvertismentService

config = Config.get()


class Dependencies:
    __chatgpt = None
    __scraper = None
    __advertisment_service = None

    @staticmethod
    def make_chatgpt() -> ChatGPT:
        if not Dependencies.__chatgpt:
            openai.api_key = Config.get()["openai_api_key"]
            Dependencies.__chatgpt = ChatGPT(openai)
        return Dependencies.__chatgpt

    @staticmethod
    def make_scraper() -> FacebookGroupsScraper:
        if not Dependencies.__scraper:
            Dependencies.__scraper = FacebookGroupsScraper(
                scraper=facebook_scraper,
                config={
                    "pages": config["scraper_pages"],
                    "posts_per_pages": config["scraper_posts_per_pages"],
                    "timeout": config["scraper_timeout"],
                },
            )
        return Dependencies.__scraper

    @staticmethod
    def make_advertisment_service() -> AdvertismentService:
        if not Dependencies.__advertisment_service:
            Dependencies.__advertisment_service = AdvertismentService(
                scraper=Dependencies.make_scraper(),
                chatGPT=Dependencies.make_chatgpt(),
            )
        return Dependencies.__advertisment_service
