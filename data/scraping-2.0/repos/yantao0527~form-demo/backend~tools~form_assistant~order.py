import logging
import json

from typing import Any, Optional, Type

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools import BaseTool

logger = logging.getLogger(__name__)

class WorkOrderFormInput(BaseModel):
    work_type: str = Field(description="Type of work, divided into carpentry and bricklaying")
    description: str = Field(description="Work order content description")
    budget: str = Field(description="Max budget")

class WorkOrderForm(BaseTool):
    name: str = "work_order_form"
    description: str = "Fill work order form"
    args_schema: Type[BaseModel] = WorkOrderFormInput

    def _run(self,
        work_type: str,
        description: str,
        budget: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        logger.info("WorkOrderForm")
        logger.info(work_type)
        logger.info(description)
        logger.info(budget)
        return json.dumps({
            "message": "created work order form OK"
        }, indent=4)



class PurchaseOrderFormInput(BaseModel):
    goods: str = Field(description="Building material name")
    unit: str = Field(description="Building material unit")
    quantity: str = Field(description="Purchase quantity")

class PurchaseOrderForm(BaseTool):
    name: str = "purchase_order_form"
    description: str = "Purchase building materials"
    args_schema: Type[BaseModel] = PurchaseOrderFormInput

    def _run(self,
        goods: str,
        unit: str,
        quantity: int,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        logger.info("PurchaseOrderForm")
        logger.info(goods)
        logger.info(unit)
        logger.info(quantity)
        return json.dumps({
            "message": "created purchase order form OK"
        }, indent=4)

