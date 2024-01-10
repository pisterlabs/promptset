from uuid import UUID

import gotrue
from fastapi import APIRouter, Depends, HTTPException
from openai.types import ImagesResponse
from pymongo.database import Database

from autobots.action.action.action_doc_model import ActionDoc, ActionCreate
from autobots.action.action.common_action_models import TextObj
from autobots.action.action.user_actions import UserActions
from autobots.action.action_type.action_text2img.action_text2img_dalle_openai_v2 import ActionCreateGenImageDalleOpenai, \
    ActionGenImageDalleOpenAiV2
from autobots.auth.security import get_user_from_access_token
from autobots.conn.openai.openai_images.image_model import ImageReq
from autobots.core.database.mongo_base import get_mongo_db
from autobots.core.logging.log import Log
from autobots.user.user_orm_model import UserORM

router = APIRouter()


@router.post("/text2img/dalle_openai")
async def create_text2img_dalle_openai(
        action_create: ActionCreateGenImageDalleOpenai,
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


@router.post("/text2img/dalle_openai/{action_id}/run")
async def run_text2img_dalle_openai(
        action_id: str,
        action_input: TextObj,
        user_res: gotrue.UserResponse = Depends(get_user_from_access_token),
        db: Database = Depends(get_mongo_db)
) -> ImagesResponse:
    try:
        user_orm = UserORM(id=UUID(user_res.user.id))
        action = await UserActions(user=user_orm, db=db).get_action(action_id)
        text2img = ActionGenImageDalleOpenAiV2(
            ImageReq.model_validate(action.config)
        )
        resp = await text2img.run_action(action_input)
        return resp
    except Exception as e:
        Log.error(str(e))
        raise HTTPException(500)
