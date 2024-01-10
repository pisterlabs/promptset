from langchain.chains import create_extraction_chain
from langchain.chat_models import ChatOpenAI

schema = {
    "properties": {
        "出租方": {"type": "string"},
        "出租方地址": {"type": "string"},
        "承租方": {"type": "string"},
        "承租方地址": {"type": "string"},
        "物业地址": {"type": "string"},
        "租金支付账号": {"type": "string"},
        "租金金额": {"type": "string"},
        "租赁期开始日期": {"type": "string"},
        "租赁期结束时间": {"type": "string"},
        "押金金额": {"type": "string"},
    },
    "required": ["出租方", "出租方地址", "承租方", "承租方地址", "物业地址", "租金支付账号", "租金金额", "租赁期开始日期", "租赁期结束时间", "押金金额"],
}


def get_extract_chain(model):
    llm = ChatOpenAI(temperature=0, model=model)
    chain = create_extraction_chain(schema, llm)
    return chain
