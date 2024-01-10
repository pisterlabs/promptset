import uuid

import pytest
from openai.types.chat import ChatCompletionUserMessageParam
from pymongo.database import Database

from autobots.action.action.action_doc_model import ActionDoc, ActionCreate
from autobots.action.action_market.user_actions_market import UserActionsMarket
from autobots.action.action_type.action_text2text.action_text2text_llm_chat_openai_v2 import ActionCreateText2TextLlmChatOpenai
from autobots.action.action.common_action_models import TextObj
from autobots.action.action.user_actions import UserActions
from autobots.action_graph.action_graph_result.user_action_graph_result import UserActionGraphResult
from autobots.conn.openai.openai_chat.chat_model import ChatReq
from autobots.core.utils import gen_random_str
from autobots.core.database.mongo_base import get_mongo_db
from autobots.action_graph.action_graph.action_graph_doc_model import ActionGraphCreate
from autobots.action_graph.action_graph.user_action_graph import UserActionGraphs
from autobots.event_result.event_result_model import EventResultStatus
from autobots.user.user_orm_model import UserORM


@pytest.mark.asyncio
async def test_user_graph_run_happy_path(set_test_settings):
    rand: str = gen_random_str()

    user_id = uuid.UUID("4d5d5063-36fb-422e-a811-cac8c2003d38")
    user = UserORM(id=user_id)

    db = next(get_mongo_db())

    user_actions = UserActions(user=user, db=db)
    user_action_market = UserActionsMarket(user, db)
    user_action_graph = UserActionGraphs(user=user, db=db)
    user_action_graph_result = UserActionGraphResult(user, db)

    # create actions
    action_llm_persona = await create_action_persona(user_actions, db, rand)
    action_llm_manager = await create_action_manager(user_actions, db, rand)
    action_llm_product = await create_action_product(user_actions, db, rand)
    action_llm_creative = await create_action_creative(user_actions, db, rand)
    action_llm_jingle = await create_action_jingle(user_actions, db, rand)

    try:
        node_map = {
            "n1": str(action_llm_manager.id),
            "n2": str(action_llm_persona.id),
            "n3": str(action_llm_product.id),
            "n4": str(action_llm_creative.id),
            "n5": str(action_llm_jingle.id)

        }
        # create action graph
        action_graph = {
            "n1": ["n2", "n3"],
            "n2": ["n4", "n5"],
            "n3": ["n4", "n5"]
        }

        # create action_graph
        action_graph_create = ActionGraphCreate(
            name="Marketing Dept",
            nodes=node_map,
            graph=action_graph
        )
        action_graph_doc = await user_action_graph.create(action_graph_create, db)

        # run action graph
        user_input = TextObj(text="Campaign for Nike shoes during Diwali Festival")
        action_graph_result_doc = await user_action_graph.run_in_background(
            user_actions,
            user_action_market,
            user_action_graph_result,
            action_graph_doc.id,
            user_input,
            None
        )

        assert action_graph_result_doc.status == EventResultStatus.success

        # user_input = TextObj(input="Campaign for Nike shoes during Diwali Festival")
        # action_graph_response = await ActionGraph.run(user, user_input, node_map, action_graph, db)
        # assert len(action_graph_response) > 1
        #
        # # save action graph
        # action_graph_create = ActionGraphCreate(
        #     name="Marketing Dept", graph=action_graph
        # )
        # action_graph_doc = await user_action_graph.create(action_graph_create, db)
        #
        # # run saved action graph
        # input = TextObj(input="Create ad for sports shoes")
        # action_graph_resp = await user_action_graph.run(action_graph_doc.id, input, db)
        # assert len(action_graph_resp) > 1

        # cleanup action graph
        await user_action_graph_result.delete_action_graph_result(action_graph_doc.id)
        await user_action_graph.delete(action_graph_doc.id, db)

    except Exception as e:
        assert e is not None

    finally:
        # cleanup actions
        await user_actions.delete_action(action_llm_manager.id)
        await user_actions.delete_action(action_llm_persona.id)
        await user_actions.delete_action(action_llm_product.id)
        await user_actions.delete_action(action_llm_creative.id)
        await user_actions.delete_action(action_llm_jingle.id)


@pytest.mark.asyncio
async def create_action_persona(user_actions: UserActions, db: Database, rand: str) -> ActionDoc:
    action_create = ActionCreateText2TextLlmChatOpenai(
        name="persona_" + rand,
        config=ChatReq(messages=[ChatCompletionUserMessageParam(
            role="user",
            content="Generate personas for Marketing this product"
        )])
    )
    action_doc = await user_actions.create_action(
        ActionCreate(**action_create.model_dump())
    )
    return action_doc


@pytest.mark.asyncio
async def create_action_manager(user_actions: UserActions, db: Database, rand: str) -> ActionDoc:
    action_create = ActionCreateText2TextLlmChatOpenai(
        name="manager_" + rand,
        config=ChatReq(messages=[ChatCompletionUserMessageParam(
            role="user",
            content="Act as market manager, create input for department"
        )])
    )
    action_doc = await user_actions.create_action(
        ActionCreate(**action_create.model_dump())
    )
    return action_doc


@pytest.mark.asyncio
async def create_action_product(user_actions: UserActions, db: Database, rand: str) -> ActionDoc:
    action_create = ActionCreateText2TextLlmChatOpenai(
        name="market researcher_" + rand,
        config=ChatReq(messages=[ChatCompletionUserMessageParam(
            role="user",
            content="Act as product researcher, create research report for the product"
        )])
    )
    action_doc = await user_actions.create_action(
        ActionCreate(**action_create.model_dump())
    )
    return action_doc


@pytest.mark.asyncio
async def create_action_creative(user_actions: UserActions, db: Database, rand: str = gen_random_str()) -> ActionDoc:
    action_create = ActionCreateText2TextLlmChatOpenai(
        name="creative_" + rand,
        config=ChatReq(messages=[ChatCompletionUserMessageParam(
            role="user",
            content="Act as a creative editor, generate text creative"
        )])
    )
    action_doc = await user_actions.create_action(
        ActionCreate(**action_create.model_dump())
    )
    return action_doc


@pytest.mark.asyncio
async def create_action_jingle(user_actions: UserActions, db: Database, rand: str = gen_random_str()) -> ActionDoc:
    action_create = ActionCreateText2TextLlmChatOpenai(
        name="jingle_" + rand,
        config=ChatReq(messages=[ChatCompletionUserMessageParam(
            role="user",
            content="Act as a creative editor, generate jingle for marketing"
        )])
    )
    action_doc = await user_actions.create_action(
        ActionCreate(**action_create.model_dump())
    )
    return action_doc

# @pytest.mark.asyncio
# async def create_graph(user_graphs: UserGraphs, graph_map: Dict[str, List[str]], db: Session, rand: str = gen_random_str()) -> GraphORM:
#     user_graph_create_input = UserGraphCreateInput(
#         name="marketing dept_" + rand,
#         graph_map=graph_map
#     )
#     graph = await user_graphs.create(user_graph_create_input, db)
#     return graph
