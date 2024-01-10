import os
from typing import List, Type, Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic.v1 import BaseModel
from pymongo import MongoClient


class StepModel(BaseModel):
    description: str
    execution: List[str]
    expected_result: str


class TestModel(BaseModel):
    application: str
    feature: str
    creator: str
    tags: List[str]
    steps: List[StepModel]


def _save_test(test: TestModel) -> str:
    """Saves a test to the database"""
    mongo_connection_string = os.getenv("MONGO_CONNECTION_STRING")
    mongo_database = os.getenv("MONGO_DATABASE_NAME")
    collection_name = os.getenv("COLLECTION_NAME")

    client = MongoClient(mongo_connection_string)

    db = client[mongo_database]
    collection = db[collection_name]
    # convert test to dict
    test_dict = test.dict()
    resp = collection.insert_one(test.dict())
    return "Test saved"


async def _asave_test(test: TestModel) -> str:
    """Saves a test to the database"""
    return _save_test(test)


class SaveTestTool(BaseTool):
    name = "save_test"
    description = "Usefull when you need to save a test"
    args_schema: Type[TestModel] = TestModel

    def _run(
            self,
            application: str,
            feature: str,
            creator: str,
            tags: List[str],
            steps: List[StepModel],
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        test = TestModel(
            application=application,
            feature=feature,
            creator=creator,
            tags=tags,
            steps=steps,
        )
        return _save_test(test)

    async def _arun(
            self,
            application: str,
            feature: str,
            creator: str,
            tags: List[str],
            steps: List[StepModel],
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        test = TestModel(
            application=application,
            feature=feature,
            creator=creator,
            tags=tags,
            steps=steps,
        )
        return await _asave_test(test)


save_test_tool = SaveTestTool()
