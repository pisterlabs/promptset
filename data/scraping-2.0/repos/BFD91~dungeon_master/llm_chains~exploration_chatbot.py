from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory, ConversationSummaryMemory, CombinedMemory, ConversationEntityMemory

from llm_chains.helpers.llms import gpt_3_5


template = """{scene}

Add the Dungeon Master's response to the player's action (either narrating what happens, taking the role of an NPC in \
a dialogue depending on the situation, or asking the player to do an ability check). The plan is that the scene above is being acted out during a DnD session, but the \
players may steer the story in a different direction, in which case the DM shall be flexible and improvise, but gently offer the player \
opportunities to navigate back to the original story. The DM's response should follow the DnD 5e rules, and ask the player to roll dice \
for the appropriate ability or skill checks in situations where the player attempts some non-trivial action. (The following \
are some examples where an ability check is appropriate: If the player attempts to kick in a door ask the player to do an \
 athletics check. If the player attempts to haggle ask the player to do a persuation check. If the player attempts to spot \
 magical traps ask the player to do an arcana check. Etc.) The DM should narrate vividly and with immersion.\
If at the very beginning of the DM and player back-and-forth, we are still at the very beginning of the scene, so respond accordingly! \
If a fight starts, print the text "~Fight~" verbatim and then stop. \
If the end of the scene is reached and it's time for the next scene to begin, print the text "~Next Scene~" verbatim and then stop. \
If the end of the entire adventure is reached, print the text "~End Adventure~" verbatim and then stop. \

Some information about characters and other entities in the scene: {entities}

The game played up so far: {history}

Player input: {player_input}
DM, either narrating the story, roleplaying some NPC, or asking the player to do an ability check, but not taking any action on behalf of the player:"""

exploration_prompt = PromptTemplate(
    input_variables=["scene", "entities", "history", "player_input"],
    template=template
)



