from typing import List, Tuple, Dict, Any

from langchain.schema import BaseOutputParser


class EntityAlignOutputParser(BaseOutputParser):
    """Parse out comma separated lists."""

    @property
    def lc_serializable(self) -> bool:
        return True

    def get_format_instructions(self) -> str:
        return (
            "If you think the candidate entity 1 and 3 describe "
            "the same concept as the source entity, you should "
            "return `[1 <name of entity 1>,3 <name of entity 3>]`. "
            "If you think none of the candidate entities describe "
            "the same concept as the source entity, you should "
            "return `[]`"
        )

    def parse(self, text: str) -> List[Tuple]:
        """Parse the output of an LLM call."""
        entity_pair_list = []
        if text != "[]":
            entity_list = text[1:-1].strip().split(", ")
            entity_pair_list = [(int(entity.split(" ")[0]), " ".join(entity.split(" ")[1:])) for entity in entity_list]
        return entity_pair_list


class EntityMergeOutputParser(BaseOutputParser):
    """Parse out comma separated lists."""

    @property
    def lc_serializable(self) -> bool:
        return True

    def get_format_instructions(self) -> str:
        return "The result should be formatted as `<name>: <description> generalbility: <Ture|False>`."

    def parse(self, text: str) -> Dict[str, Any]:
        """Parse the output of an LLM call.

        :param text: the output of an LLM call
        :type text: str
        :return: the name
        :rtype: str
        """
        name, description_generalbility = text.split(": ", 1)
        description, generalbility_str = description_generalbility.split(" generalbility: ")
        if 'true' in generalbility_str.lower():
            generalbility = True
        elif 'false' in generalbility_str.lower():
            generalbility = False
        else:
            raise ValueError(f"generalbility_str should be either True or False, but got {generalbility_str}")
        return {"name": name, "description": description, "general": generalbility}
