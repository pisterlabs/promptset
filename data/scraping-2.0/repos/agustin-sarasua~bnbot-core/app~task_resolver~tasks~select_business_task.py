from typing import List, Optional
from app.task_resolver.engine import Task, Step
from app.task_resolver.engine.task_model import Step
from app.task_resolver.step_resolvers import GatherBusinessInfoResolver, BusinessSelectionResolver, PostProcessRouterResolver
from app.integrations import OpenAIClient
from .make_reservation_task import create_make_reservation_task

class SelectBusinessTask(Task):

    def __init__(self):
        gather_business_info_step = Step(name="GATHER_BUSINESS_INFO", 
                                         resolver =GatherBusinessInfoResolver(), 
                                         reply_when_done=False)
        
        business_selection_step = Step(name="BUSINESS_SELECTION", 
                                       resolver = BusinessSelectionResolver(backend_url="http://web:80"), 
                                       reply_when_done=False)
        
        super().__init__(name="SELECT_BUSINESS_TASK", 
                         steps=[gather_business_info_step, business_selection_step])

    def get_next_task(self) -> Optional[Task]:
        if self.is_done():
            if (self.steps[-1].data.resolver_data["business_info"]["bnbot_id"] is not None and
                self.steps[-1].data.resolver_data["business_info"]["bnbot_id"] != ""):
                return create_make_reservation_task()
        return None
    
def create_select_business_task():
    return SelectBusinessTask()