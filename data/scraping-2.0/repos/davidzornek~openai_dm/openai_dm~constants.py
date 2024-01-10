# flake8: noqa
from openai_dm.tools import (
    AbilityScoreTool,
    BackgroundTool,
    RaceTool,
    ClassTool,
    CharacterSheetInspector,
    SkillProficiencydTool,
)

COST_PER_1000_TOKENS = {
    "text-davinci-003": {
        "input": 0.002,
        "output": 0.002,
    },
    "gpt-3.5-turbo": {
        "input": 0.0015,
        "output": 0.002,
    },
    "gpt-4": {
        "input": 0.03,
        "output": 0.06,
    },
}

NODE_RULES = {
    "race": [
        "After the player indicates their choice of race, use the appropriate tool to update their character sheet.",
    ],
    "class_": [
        "After the player indicates their choice of class, use the appropriate tool to update their character sheet.",  # noqa: E501
    ],
    "ability_scores": [
        """Do the following:
        1. Find out which method the user wants for generating ability scores: standard array or point buy.
        2. Offer to optimize the ability scores for the user's class, which can be obtained from the character sheet.
        2a. If you optimize ability scores for the player, ask permission before updating the character sheet.
        3. Use the appropriate tool to update the character sheet with the new ability scores."""
    ],
    "background": [
        "After the player indicates their choice of background, use the appropriate tool to update their character sheet.",  # noqa: E501
    ],
    "skill_proficiencies": [
        """Do the following;
        1. Check which skill proficiences the player already has from their background.
        2. Check which skill proficiences the player's class allows them to choose from.
        3. Assist the player in choosing additional skill proficiencies they don't already have.
        4. Update the character sheet with new skill proficiencies.""",
        "When listing options for the player, only include skill proficiencies offered by their character's class_",
    ],
}

NODE_TOOLS = {
    "race": [RaceTool, CharacterSheetInspector],
    "class_": [ClassTool, CharacterSheetInspector],
    "ability_scores": [AbilityScoreTool, CharacterSheetInspector],
    "background": [BackgroundTool, CharacterSheetInspector],
    "skill_proficiencies": [SkillProficiencydTool, CharacterSheetInspector],
}
