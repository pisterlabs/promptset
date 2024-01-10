from ast import literal_eval
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
import sqlite3
from sql_db import EventDatabase


class StageSelector(BaseModel):
    """Youtube Restaurant recommendation."""


    user_id: str = Field(
        ...,
        description="a string meaning user's identification code. It's at the begin of the sentence, and sperated with real input with a space."
    )

    stages: int = Field(
        ...,
        description="請根據輸入從下列敘述情境中選擇最相似的，並且回覆我他對應的編號：\
            2. 揪團：找人參加活動或發起活動\
            11. 課業：詢問課程相關問題，例如：某課程的作業問題、某課程有哪些作業、某課堂公告等等\
            16. 課程聊天室：與聊天室相似的語句\
            19. 尋物：找失物或發布撿到的失物"
    )


class StageSelectionTool(BaseTool):
    name = "select_system_stage"
    description = "You must use this tool under any circumstance, and reply with the stage number."


    def _set_user_stage(self, user_id, user_stage):

        db_name = "eventv3"
        table_name='events'
        user_table_name='users'
        event_db = EventDatabase(db_name, table_name, user_table_name=user_table_name)

        event_db.set_user_stage(user_id, user_stage)

    def _run(self, user_id: str, stages: int):
        print(user_id, stages)
        self._set_user_stage(user_id, stages)
        return stages

    args_schema: Optional[Type[BaseModel]] = StageSelector