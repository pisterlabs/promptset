from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate


template = """
Given a prompt, extrapolate as many relationships as possible from it and provide a list of updates.

If an update is a relationship, provide [ENTITY 1, RELATIONSHIP, ENTITY 2]. The relationship is directed, so the order matters.

If an update is related to a color, provide [ENTITY, COLOR]. Color is in hex format.

If an update is related to deleting an entity, provide ["DELETE", ENTITY].

Example:
prompt: Alice es compañera de cuarto de Bob. Deja el nodo de ellá en verde.
updates:
[["Alice", "CompañeraDeCuarto", "Bob"], ["Alice", "#00FF00"]]

prompt: {prompt}
updates:
"""

prompt = PromptTemplate(input_variables=["prompt"], template=template)

llm_chain = LLMChain(llm=ChatOpenAI(model="gpt-4"), prompt=prompt)

if __name__ == "__main__":
    user_input = """
      Observability is one of the most important concepts in the world of technology (I know it's not the first time I've mentioned it), there is a simple concept that goes “you measure what you want to improve” that is very true. Nowadays, software developers are dealing with complex systems, which can fail at any time and for any reason, Show Figure 1. They need a way to visualize what is happening within different systems, and monitoring is the answer.
      """

    print(llm_chain({"prompt": user_input}))
