from langchain.prompts import StringPromptTemplate
from langchain.agents import Tool

from typing import List


class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


english_agent_template = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


french_agent_template = """
Répondez aux questions suivantes de la manière la plus détaillée possible. Si la question implique une comparaison, veuillez fournir une liste détaillée des différences. Vous avez accès aux outils suivants :

{tools}

Utilisez le format suivant :

Question: la question à laquelle vous devez répondre
Réflexion: vous devriez toujours réfléchir à quoi faire
Action: Le nom de l'outil. l'outil est contraint d'être un dans la sélection suivante : [{tools}]
Paramètres: Paramètres de l'outil
Observation: Le résultat de l'action
... (ce cycle Réflexion/Action/Paramètres/Observation peut se répéter N fois)
Réflexion: Je connais maintenant la réponse finale
Réponse Finale: la réponse finale à la question initiale


Commencez!

Question: {input}
Réflexion: {agent_scratchpad}"""