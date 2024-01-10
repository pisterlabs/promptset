import math

from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session
import openai
import json
from dotenv.main import load_dotenv
from os import environ

load_dotenv()

openai.api_key = environ["API_KEY"]

class CRUD:
    def __init__(self, session: Session) -> None:
        self.session = session
        # self.query = self.session.query(table)

    def get_list(self, table: BaseModel):
        return self.session.query(table).all()

    def get_record(self, table: BaseModel, cond={}):
        filters = []
        for table_id, id in cond.items():
            filters.append(getattr(table, table_id) == id)
        return self.session.query(table).filter(*filters).first()

    def create_record(self, table: BaseModel, req: BaseModel):
        db_record = table(**req.dict())
        print(db_record.ans_1)
        self.session.add(db_record)
        self.session.commit()
        self.session.refresh(db_record)
        return db_record

    def update_record(self, db_record: BaseModel, req: BaseModel):
        req = req.dict()
        for key, value in req.items():
            setattr(db_record, key, value)
        self.session.commit()

        return db_record

    def patch_record(self, db_record: BaseModel, req: BaseModel):
        req = req.dict()
        for key, value in req.items():
            if value:
                setattr(db_record, key, value)
            if value == 0:
                setattr(db_record, key, value)
        self.session.commit()

        return db_record

    def delete_record(self, table: BaseModel, cond={}):
        db_record = self.get_record(table, cond)
        if db_record:
            self.session.delete(db_record)
            self.session.commit()
            return 1
        else:
            return -1

    def paging_record(self, table: BaseModel, req: BaseModel):
        total_row = self.session.query(table).count()
        if total_row % req.size == 0:
            total_page = math.floor(total_row / req.size)
        else:
            total_page = math.floor(total_row / req.size) + 1
        start = (req.page - 1) * req.size

        items = self.session.query(table).order_by(table.create_time.desc()).offset(start).limit(req.size).all()
        pages = {"items": items, "total_pages": total_page, "page": req.page, "size": req.size, "total_row": total_row}
        return pages

    def search_record(self, table: BaseModel, req: BaseModel):
        req_dict = req.dict()
        filters = []
        for key, value in req_dict.items():
            if value == 0 or value:
                if isinstance(value, (int, float)):
                    filters.append(getattr(table, key) == value)
                elif isinstance(value, str):
                    filters.append(getattr(table, key).contains(value))
                elif isinstance(value, list):
                    filters.append(func.json_contains(getattr(table, key), str(value)) == 1)

        result = self.session.query(table).filter(*filters).all()
        return result
    
    def ai_create_record(self, table: BaseModel, req: BaseModel):
        db_record = table(**req.dict())
        content = db_record.ans_1
        situation = db_record.ans_1 + db_record.ans_3
        completion = openai.ChatCompletion.create(
        model = 'gpt-3.5-turbo',
        messages = [
            {
                "role": "system",
                "content": "당신은 텍스트를 카테고리로 분류하는 로봇입니다. 텍스트가 어떤 범주에 속하는지 분류해야 합니다. 글의 카테고리는 다음과 같습니다. 각 카테고리는 ,로 구분되어 있습니다.\
                        [고민상담, 마음의 상처, 이별/실연, 속마음, 학업 스트레스, 직장 스트레스, 친구 갈등, 가족 갈등, 연애/결혼 문제, 우울함, 번아웃]\
                        단 하나의 카테고리만 답변하세요. 어떠한 설명이나 단어도 붙이지 마세요.",
            },
            {
                "role": "user",
                "content": content,
            },
        ],
        )

        keyword = completion['choices'][0]['message']['content']
        
        completion_sit = openai.ChatCompletion.create(
            model = 'gpt-3.5-turbo',
            messages = [
                {
                    "role": "system",
                    "content": "당신은 위로를 해주는 상담가입니다. 당신은 문제 상황에 대한 서술인 'message'의 내용과, 고민의 카테고리인 'category'에 맞는 감정적인 위로를 작성해주어야 합니다. 단, 해결책을 제시해주지 마세요. 위로는 최대 3문장으로 작성해주세요. 부드러운 말투로 작성해주세요.",
                },
                {
                    "role": "user",
                    "content": f"'message':'{situation}',\
                    'category':{keyword}",
                },
            ],
            temperature=0.6,
        )

        touch = completion_sit['choices'][0]['message']['content']
        db_record.keyword = keyword
        db_record.touch = touch
        self.session.add(db_record)
        self.session.commit()
        self.session.refresh(db_record)
        return db_record
    