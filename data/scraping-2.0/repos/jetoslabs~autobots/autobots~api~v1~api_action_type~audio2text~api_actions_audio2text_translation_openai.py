from typing import Optional
from uuid import UUID

import gotrue
from fastapi import APIRouter, Depends, HTTPException
from openai.types.audio import Translation
from pymongo.database import Database

from autobots.action.action.action_doc_model import ActionCreate, ActionDoc
from autobots.action.action.user_actions import UserActions
from autobots.action.action_type.action_audio2text.action_audio2text_translation_openai import AudioRes
from autobots.action.action_type.action_types import ActionType
from autobots.auth.security import get_user_from_access_token
from autobots.conn.openai.openai_audio.translation_model import TranslationReq
from autobots.core.database.mongo_base import get_mongo_db
from autobots.core.logging.log import Log
from autobots.user.user_orm_model import UserORM

router = APIRouter()


class ActionCreateAudio2TextTranslationOpenai(ActionCreate):
    type: ActionType = ActionType.audio2text_translation_openai
    config: TranslationReq
    input: Optional[AudioRes] = None
    output: Optional[Translation] = None


@router.post("/audio2text/translation_openai")
async def create_action_audio2text_translation_openai(
        action_create: ActionCreateAudio2TextTranslationOpenai,
        user_res: gotrue.UserResponse = Depends(get_user_from_access_token),
        db: Database = Depends(get_mongo_db)
) -> ActionDoc:
    try:
        user_orm = UserORM(id=UUID(user_res.user.id))
        action_doc = await UserActions(user_orm, db).create_action(
            ActionCreate(**action_create.model_dump())
        )
        return action_doc
    except Exception as e:
        Log.error(str(e))
        raise HTTPException(500)
