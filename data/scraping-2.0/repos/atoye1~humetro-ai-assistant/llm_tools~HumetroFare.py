from pydantic.v1 import BaseModel, Field
from langchain.agents import tool
from typing import List


# TODO: add better filters
class HumetroFaresInput(BaseModel):
    ages: List[str] = Field(
        ..., description="list of extracted age of the user. element must be one of 유아, 어린이, 청소년, 성인, 다자녀, 장애인, 국가유공자, 만65세 이상, 전체 if not specified, pass [전체]")


@tool(args_schema=HumetroFaresInput)
def get_fares(ages: List[str]) -> str:
    """You MUST you this tool when answering questions about fare(요금)"""
    result = ""
    if "유아" in ages:
        result += "유아(만5세 이하) : 보호자 1명당 최대 3인까지 무료입니다."
    if "어린이" in ages:
        result += "어린이의(만6세 이상 ~ 만12세 이하, 초등학생) : 교통카드와 모바일승차권은 1구간과 2구간 모두 무료이며, QR승차권은 1구간 700원, 2구간 800원입니다."
    if "청소년" in ages:
        result += "청소년(만13세 이상 ~ 만18세 이하, 중학생, 고등학생) : 교통카드와 모바일승차권은 1구간 1,050원, 2구간 1,200원이며, QR승차권은 카드에서 100원이 추가된 1구간 1,150원, 2구간 1,300원."
    if "성인" in ages:
        result += "성인(만19세이상 ~ 만64세 이하, 대학생, 모든 외국인): 교통카드와 모바일승차권은 1구간 1,450원, 2구간 1,650원이며, QR승차권은 카드에서 100원이 추가된 1구간 1,550원, 2구간 1,750원."
    if "다자녀" in ages:
        result += "다자녀( ): 교통카드는 1구간 750원, 2구간 850원이며, QR승차권은 1구간 800원, 2구간 900원입니다."
    if "장애인" in ages:
        result += "만65세 이상, 장애인, 국가유공자는 유효한 신분증이나 카드를 소지한 경우 1구간과 2구간모두 무료입니다."
    if "국가유공자" in ages:
        result += "만65세 이상, 장애인, 국가유공자는 유효한 신분증이나 카드를 소지한 경우 1구간과 2구간모두 무료입니다."
    if "만65세 이상" in ages:
        result += "만65세 이상, 장애인, 국가유공자는 유효한 신분증이나 카드를 소지한 경우 1구간과 2구간모두 무료입니다."
    else:
        result = """
유아(만5세 이하) : 보호자 1명당 최대 3인까지 무료입니다. 
어린이의(만6세 이상 ~ 만12세 이하, 초등학생) : 교통카드와 모바일승차권은 1구간과 2구간 모두 무료이며, QR승차권은 1구간 700원, 2구간 800원입니다.
청소년(만13세 이상 ~ 만18세 이하, 중학생, 고등학생) : 교통카드와 모바일승차권은 1구간 1,050원, 2구간 1,200원이며, QR승차권은 카드에서 100원이 추가된 1구간 1,150원, 2구간 1,300원.
성인(만19세이상 ~ 만64세 이하, 대학생, 모든 외국인): 교통카드와 모바일승차권은 1구간 1,450원, 2구간 1,650원이며, QR승차권은 카드에서 100원이 추가된 1구간 1,550원, 2구간 1,750원.
다자녀(주소지가 부산이며 세자녀 이상이고 막내가 만18세 이하.): 교통카드는 1구간 750원, 2구간 850원이며, QR승차권은 1구간 800원, 2구간 900원입니다. 다자녀 할인은 동해남부선에 적용되지 않습니다.
만 65세 이상의 노인, 장애인 복지카드, 국가유공자는 유효한 신분증이나 카드를 소지한 경우 1구간과 2구간모두 무료입니다.
"""

    return result
