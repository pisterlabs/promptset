from fastapi import APIRouter, HTTPException, status
from langchain_core.messages import HumanMessage

from schemas.ml import MlResponse, MlRequest
from langchain.chat_models.gigachat import GigaChat

router = APIRouter(prefix="/api/v1/ml", tags=["ml"])


@router.post(
    "/",
)
async def post(req: MlRequest):
    if req.description == "":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="There is not description")

    res = {}

    chat = GigaChat(credentials="Mzk3MTJlMGEtMTkyZS00YWE0LWJmMGEtMzRlNzE1ZmIzZGNjOmI1NzA1YmFmLTQyNTctNDdjMy1hNzdkLWI0ZTQyZTM5ZGY4Ng==",
                    verify_ssl_certs=False, temperature=1.0)

    prompt = "Ты профессиональный бизнес аналитик и экономист. Распиши бизнес модель по описанию проекта: "
    ans = chat([HumanMessage(content=prompt + req.description)])
    res["Бизнес модель"] = ans.content

    prompt = "Ты профессиональный бизнес аналитик и экономист. Распиши экономику по описанию проекта: "
    ans = chat([HumanMessage(content=prompt + req.description)])
    res["Экономика"] = ans.content

    prompt = "Ты профессиональный бизнес аналитик и экономист. Распиши конкуренты по описанию проекта: "
    ans = chat([HumanMessage(content=prompt + req.description)])
    res["Конкуренты"] = ans.content

    prompt = "Ты профессиональный бизнес аналитик и экономист. Распиши план развития по описанию проекта: "
    ans = chat([HumanMessage(content=prompt + req.description)])
    res["План развития"] = ans.content

    prompt = "Ты профессиональный бизнес аналитик и экономист. Распиши ценностное предложение по описанию проекта: "
    ans = chat([HumanMessage(content=prompt + req.description)])
    res["ценностное предложение"] = ans.content

    prompt = "Ты профессиональный бизнес аналитик и экономист. Распиши команду по описанию проекта: "
    ans = chat([HumanMessage(content=prompt + req.description)])
    res["команда"] = ans.content

    return res
