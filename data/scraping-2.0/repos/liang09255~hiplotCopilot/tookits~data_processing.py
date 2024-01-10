from typing import List
from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.tools import BaseTool
from tools import discrete_mapping, missing_padding


class DataProcessingToolKit(BaseToolkit):
    def get_tools(self) -> List[BaseTool]:
        return [
            discrete_mapping.DiscreteMapping(),
            missing_padding.MissingPadding(),
        ]