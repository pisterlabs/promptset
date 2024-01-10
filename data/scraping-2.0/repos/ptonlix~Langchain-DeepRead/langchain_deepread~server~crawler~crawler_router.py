import logging
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from langchain_deepread.server.utils.auth import authenticated
from langchain_deepread.components.article_crawler import Article
from langchain_deepread.server.crawler.crawler_service import CrawlerService
from langchain_deepread.server.utils.model import RestfulModel

logger = logging.getLogger(__name__)

crawler_router = APIRouter(prefix="/v1", dependencies=[Depends(authenticated)])


class CrawlerBody(BaseModel):
    url: str

    model_config = {
        "json_schema_extra": {
            "examples": [{"url": "https://mp.weixin.qq.com/s/a15uF6hC5aDvZKuVLhwFEQ"}]
        }
    }


@crawler_router.post(
    "/cralwer",
    response_model=RestfulModel[Article | None],
    tags=["Crawler"],
)
def crawler(request: Request, body: CrawlerBody) -> RestfulModel:
    """
    获取文章内容
    """
    service = request.state.injector.get(CrawlerService)
    return RestfulModel(data=service.crawler(body.url))
