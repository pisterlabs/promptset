import pytest
from openai.types.chat import ChatCompletionUserMessageParam

from autobots.action.action.action_crud import ActionCRUD
from autobots.action.action.action_doc_model import ActionDocCreate, ActionDocFind
from autobots.action.action.common_action_models import TextObj
from autobots.action.action_result.user_action_result import UserActionResult
from autobots.action.action_type.action_factory import ActionFactory
from autobots.action.action_type.action_types import ActionType
from autobots.conn.openai.openai_chat.chat_model import ChatReq
from autobots.core.database.mongo_base import get_mongo_db
from autobots.core.utils import gen_uuid
from autobots.user.user_orm_model import UserORM


@pytest.mark.asyncio
async def test_action_mock_happy_path(set_test_settings):
    db = next(get_mongo_db())
    action_crud = ActionCRUD(db)
    user_id = gen_uuid()
    user_action_result = UserActionResult(UserORM(id=user_id), db)
    action_result_doc = None
    action_name = "test_action_mock_happy_path"

    try:
        action_doc_create = ActionDocCreate(
            name=action_name,
            type=ActionType.mock_action,
            config=TextObj(text="mock_action").model_dump(),
            # input=None,
            # output=None,
            user_id=str(user_id)
        )
        inserted = await action_crud.insert_one(action_doc_create)
        assert inserted is not None

        action_find = ActionDocFind(id=str(inserted.id), user_id=str(user_id))
        action_docs = await action_crud.find(action_find)
        assert len(action_docs) == 1
        action_doc = action_docs.pop()

        action_input = TextObj(text="okok")

        action_result_doc = await ActionFactory().run_action_in_background(
            action_doc,
            action_input.model_dump(),
            user_action_result
        )

    except Exception as e:
        assert e is None
    finally:
        deleted = await user_action_result.delete_action_result(action_result_doc.id)
        assert deleted == 1

        find = ActionDocFind(name=action_name, user_id=str(user_id))
        deleted = await action_crud.delete_many(find)
        assert deleted.deleted_count == 1

@pytest.mark.asyncio
async def test_action_llm_chat_happy_path(set_test_settings):
    db = next(get_mongo_db())
    action_crud = ActionCRUD(db)
    user_id = gen_uuid()
    user_action_result = UserActionResult(UserORM(id=user_id), db)
    action_result_doc = None
    action_name = "test_action_llm_chat_happy_path"

    try:
        chat_req = ChatReq(messages=[ChatCompletionUserMessageParam(role="user", content="You are an expert blogger")])
        action_doc_create = ActionDocCreate(
            name=action_name,
            type=ActionType.text2text_llm_chat_openai,
            config=chat_req.model_dump(),
            # input=None,
            # output=None,
            user_id=str(user_id)
        )
        inserted = await action_crud.insert_one(action_doc_create)
        assert inserted is not None

        action_find = ActionDocFind(id=str(inserted.id), user_id=str(user_id))
        action_docs = await action_crud.find(action_find)
        assert len(action_docs) == 1
        action_doc = action_docs.pop()

        action_input = TextObj(text="okok")

        action_result_doc = await ActionFactory().run_action_in_background(
            action_doc,
            action_input.model_dump(),
            user_action_result
        )

    except Exception as e:
        assert e is None
    finally:
        deleted = await user_action_result.delete_action_result(action_result_doc.id)
        assert deleted == 1

        find = ActionDocFind(name=action_name, user_id=str(user_id))
        deleted = await action_crud.delete_many(find)
        assert deleted.deleted_count == 1
