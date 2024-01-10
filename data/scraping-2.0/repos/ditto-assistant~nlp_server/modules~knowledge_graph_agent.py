KG_TEMPLATE = """
You are en experienced neo4j graph constructor and your job is to convert this text into nodes and relationships for a neo4j graph. Use integer ids for node ids and encode the nodes and relationships in JSON. The user's prompt that generated the text is included below. The user's prompt will always be the first node with id=0, type=Prompt, title=Prompt, and description=a summary of the user's prompt. The second node will always be a summary of the text with id=1, type=Response, title=Response, and description=a summary of the text. The rest of the nodes and relationships are up to you.

Do not include new information than what's given in Text.

Nodes have the following keys:
1. id: unique integer identifier
2. type: usually the subject of the sentence or paragraph but can be any one word that describes the node. Make this one word that summarizes header strings.
3. title: single or few word title of the node.
4. description: a description of the node. 

Relationships have the following keys:
1. src_node: id of source node.
2. target_node: id of target node.
3. relationship_type: unique relationship string that can be used if appears again. Make this all caps with underscores as spaces.

Be sure to always make a Prompt, Response, and Subject node. Always connect a node to at least one other node. Never leave a node unconnected. If you are unsure where to connect a node related to a Subject and its nodes, connect it to the Subject node by default.

Also, ALWAYS use single quotes for strings in JSON. NEVER use double quotes for strings in JSON.

Examples:

User's Prompt: What is the best Pokémon?

Text:
# The Strongest Pokémon

## Introduction
As of the last knowledge update in January 2022, Arceus stands out as one of the most formidable Pokémon in the expansive Pokémon franchise. This Legendary Pokémon holds the prestigious title of the "God of Pokémon" and is recognized for its pivotal role in the creation of the Pokémon universe.

## Legendary Status
Arceus's legendary status is attributed to its extraordinary abilities and significance within the Pokémon lore. Revered as the "God of Pokémon," it is widely acknowledged for its unparalleled power and influence.

## Base Stats and Superiority
Arceus's base stats surpass those of many other Pokémon, solidifying its reputation as a powerhouse within the Pokémon world. These impressive stats contribute to its standing as one of the strongest Pokémon in the franchise.

## Dynamic Pokémon Franchise
It is crucial to acknowledge the dynamic nature of the Pokémon franchise. Ongoing releases of new games and generations continuously introduce additional powerful Pokémon, altering the landscape of strength and capabilities.

## Factors Influencing Strength
The determination of the "strongest" Pokémon is a nuanced process, considering factors such as individual stats, movesets, and type matchups. The interplay of these elements contributes to the varying assessments of Pokémon strength.

## Staying Informed
For the latest and most accurate information on the strongest Pokémon, especially considering potential new Pokémon games or updates since January 2022, it is highly recommended to consult the latest official Pokémon sources. Staying informed ensures an up-to-date understanding of the evolving hierarchy of Pokémon strength.

Response:
{
  "nodes": [
    {
      "id": 0,
      "type": "Prompt",
      "title": "Prompt",
      "description": "What is the best Pokémon?"
    },
    {
      "id": 1,
      "type": "Response",
      "title": "Response",
      "description": "As of January 2022, Arceus is considered the strongest Pokémon, but ongoing releases may introduce new contenders, so check official Pokémon sources for the latest information."
    },
    {
      "id": 2,
      "type": "Subject",
      "title": "Pokémon",
      "description": "Pokémon"
    },
    {
      "id": 3,
      "type": "Pokémon",
      "title": "Arceus",
      "description": "Arceus's legendary status is attributed to its extraordinary abilities and significance within the Pokémon lore. Revered as the 'God of Pokémon,' it is widely acknowledged for its unparalleled power and influence."
    },
    {
      "id": 4,
      "type": "Stats",
      "title": "Base Stats",
      "description": "Arceus's base stats surpass those of many other Pokémon, solidifying its reputation as a powerhouse within the Pokémon world. These impressive stats contribute to its standing as one of the strongest Pokémon in the franchise."
    },
    {
      "id": 5,
      "type": "Explanation",
      "title": "Pokémon Franchise",
      "description": "The Pokémon franchise is constantly evolving, with new games and generations introducing new and powerful Pokémon."
    },
    {
      "id": 6,
      "type": "Explanation",
      "title": "Determining Strength",
      "description": "The determination of the 'strongest' Pokémon is a nuanced process, considering factors such as individual stats, movesets, and type matchups."
    },
    {
      "id": 7,
      "type": "Explanation",
      "title": "Staying Informed",
      "description": "For the latest and most accurate information on the strongest Pokémon, it is highly recommended to consult the latest official Pokémon sources."
    }
  ],
  "relationships": [
    {
      "src_node": 0,
      "target_node": 1,
      "relationship_type": "HAS_RESPONSE"
    },
    {
      "src_node": 1,
      "target_node": 2,
      "relationship_type": "HAS_SUBJECT"
    },
    {
      "src_node": 2,
      "target_node": 3,
      "relationship_type": "HAS_LEGENDARY_STATUS"
    },
    {
      "src_node": 3,
      "target_node": 4,
      "relationship_type": "HAS_BASE_STATS_AND_SUPERIORITY"
    },
    {
      "src_node": 2,
      "target_node": 5,
      "relationship_type": "EXISTS_IN_DYNAMIC_FRANCHISE"
    },
    {
      "src_node": 4,
      "target_node": 6,
      "relationship_type": "FACTORS_INFLUENCING_STRENGTH"
    },
    {
      "src_node": 2,
      "target_node": 7,
      "relationship_type": "REQUIRES_STAYING_INFORMED"
    }
  ]
}

User's Prompt: What is the capital of France?

Text:
# Capital of France
The user asked the AI assistant for the capital of France, and the assistant responded with the answer: Paris.

Response:
{
  "nodes": [
    {
      "id": 0,
      "type": "Prompt",
      "title": "Prompt",
      "description": "What is the capital of France?"
    },
    {
      "id": 1,
      "type": "Response",
      "title": "Response",
      "description": "The capital of France is Paris."
    },
    {
      "id": 2,
      "type": "Subject",
      "title": "France",
      "description": "France is a country located in Western Europe. It is known for its rich history, culture, and contributions to art, fashion, and cuisine."
    },
    {
      "id": 3,
      "type": "Capital",
      "title": "Paris",
      "description": "Paris is the capital and largest city of France. It is a global center for art, fashion, gastronomy, and culture."
    }
  ],
  "relationships": [
    {
      "src_node": 0,
      "target_node": 1,
      "relationship_type": "HAS_RESPONSE"
    },
    {
      "src_node": 1,
      "target_node": 2,
      "relationship_type": "HAS_SUBJECT"
    },
    {
      "src_node": 2,
      "target_node": 3,
      "relationship_type": "HAS_CAPITAL"
    }
  ]
}


User's Prompt: <!user_prompt>

Text:
<!text>

Response:
"""

from langchain.chat_models import ChatOpenAI


class KGAgent:
    def __init__(self):
        self.template = KG_TEMPLATE
        self.llm = ChatOpenAI(temperature=0.4, model_name="gpt-3.5-turbo-16k")

    def get_prompt_template(self, user_prompt, text):
        template = KG_TEMPLATE
        template = template.replace("<!user_prompt>", user_prompt)
        template = template.replace("<!text>", text)
        return template

    def construct_kg(self, user_prompt: str, text: str):
        prompt = self.get_prompt_template(
            user_prompt.replace('"', "'"), text.replace('"', "'")
        )
        res = self.llm.call_as_llm(prompt)
        return res


if __name__ == "__main__":
    agent = KGAgent()
    user_prompt = "Can you tell me about the pokemon Mewtwo?"
    text = """
    # Mewtwo

    ## Introduction
    Mewtwo is a powerful Psychic-type Pokémon that was created through genetic manipulation. It is renowned for its extraordinary psychic abilities and exceptional intelligence. This Pokémon is a clone of the legendary Pokémon Mew, and it was specifically engineered with the ambition of becoming the most dominant and formidable Pokémon in existence.

    ## Appearance
    Mewtwo possesses a sleek and humanoid physique, characterized by its vibrant purple fur. It boasts a long, elegant tail and a distinctive, heavily armored head. These physical attributes contribute to Mewtwo's imposing presence and visually distinguish it from other Pokémon.

    ## Psychic Abilities
    Mewtwo's psychic capabilities are unparalleled within the Pokémon world. It possesses an array of psychic powers, including telekinesis, telepathy, and the ability to manipulate energy. These abilities grant Mewtwo an immense advantage in battles and make it a formidable opponent.

    ## Origin and Creation
    Mewtwo's origin lies in its genetic connection to the legendary Pokémon Mew. Scientists conducted extensive genetic experiments to create a clone of Mew, resulting in the birth of Mewtwo. The aim of these experiments was to produce a Pokémon with unparalleled power and abilities.

    ## Appearances in Media
    Mewtwo has made appearances in various Pokémon games, movies, and TV shows. It is often depicted as a significant character and a formidable adversary. Its presence in these media outlets has contributed to its popularity and recognition among Pokémon enthusiasts.

    ## Legacy and Impact
    Mewtwo's status as a powerful and iconic Pokémon has solidified its place in the Pokémon franchise. Its unique abilities, captivating appearance, and intriguing backstory have made it a fan favorite and a symbol of strength and intelligence within the Pokémon universe.

    ## Ongoing Evolution
    As the Pokémon franchise continues to evolve with new games and generations, Mewtwo's role and significance may undergo further development. It is essential to stay updated with the latest official Pokémon sources to remain informed about any new information or changes regarding Mewtwo.  
    """
    res = agent.construct_kg(user_prompt, text)
    print(res)

    ## TODO: Hook this up to Ditto Memory Agent!!!! :D
    # This will allow us to construct a knowledge graph while interacting with Ditto :D
