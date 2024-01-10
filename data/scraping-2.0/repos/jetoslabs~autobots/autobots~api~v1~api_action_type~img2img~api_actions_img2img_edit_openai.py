from uuid import UUID

import gotrue
from fastapi import APIRouter, Depends, HTTPException
from openai.types import ImagesResponse
from pymongo.database import Database

from autobots.action.action.action_doc_model import ActionDoc, ActionCreate
from autobots.action.action.user_actions import UserActions
from autobots.action.action_type.action_img2img.action_img2img_edit_openai import ImageEditInput
from autobots.action.action_type.action_types import ActionType
from autobots.auth.security import get_user_from_access_token
from autobots.conn.openai.openai_images.image_model import ImageEdit
from autobots.core.database.mongo_base import get_mongo_db
from autobots.core.logging.log import Log
from autobots.user.user_orm_model import UserORM

router = APIRouter()


class ActionCreateImg2ImgEditOpenai(ActionCreate):
    type: ActionType = ActionType.text2text_search_web
    config: ImageEdit
    input: ImageEditInput | None = None
    output: ImagesResponse | None = None


@router.post("/img2img/edit_openai")
async def create_img2img_edit_openai(
        action_create: ActionCreateImg2ImgEditOpenai,
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
