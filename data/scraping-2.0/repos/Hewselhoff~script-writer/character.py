import os

from langchain import OpenAI, ConversationChain
from langchain.prompts.chat import PromptTemplate
from langchain.memory import ConversationBufferMemory


os.environ["OPENAI_API_KEY"] = "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"


class CharacterRole():
  PROMPT_GENRE_TEMPLATE = "* You are roleplaying in {genre}.".format
  PROMPT_IDENTITY_TEMPLATE = "* From here on, you are to take on the persona of {identity} in all responses.".format
  PROMPT_ACTOR_TEMPLATE = "* Your character is played by the actor {actor}.".format
  PROMPT_CONTEXT_TEMPLATE = "* Here is some additional information about your character, for context:\n{context}".format
  PROMPT_INSTRUCTIONS_TEMPLATE = \
  """* Only write responses for {identity}.
  * Do not add narration or dialogue for characters other than {identity}.""".format
  TEMPLATE = \
  """Your Role:
  {genre}
  {identity}
  {actor}
  {context}
  {instructions}
  """.format
  def __init__(self, 
               identity:str, 
               context:list, 
               genre:str, 
               actor:str=None, 
               instructions:list=None):
    self.identity = identity
    self.context = context
    self.genre = genre
    self.actor = actor
    self.instructions = instructions
    self.identity_prompt = self.PROMPT_IDENTITY_TEMPLATE(identity=identity)
    self.genre_prompt = self.PROMPT_GENRE_TEMPLATE(genre=genre)
    context_format = "\n".join(["    * {} {}".format(identity, line) for line in context])
    self.context_prompt = self.PROMPT_CONTEXT_TEMPLATE(context=context_format)
    self.actor_prompt = self.PROMPT_ACTOR_TEMPLATE(actor=actor) if actor else ""
    self.instructions_prompt = self.PROMPT_INSTRUCTIONS_TEMPLATE(identity=identity)
    if instructions:
      self.instructions_prompt += "\n" + "\n".join(["  * {}".format(line) for line in instructions])
    # Compile role prompt
    self._string = self.TEMPLATE(identity=self.identity_prompt,
                                 context=self.context_prompt,
                                 genre=self.genre_prompt,
                                 actor=self.actor_prompt,
                                 instructions=self.instructions_prompt)

  def __str__(self):
    return self._string


CharacterPromptTemplate = """{role}

Script:
{{history}}
{{input}}
{identity}:"""


class Character:
  PROMPT_TEMPLATE = \
    ("{role}     \n"
     "           \n"
     "Script:    \n"
     "{{history}}\n"
     "{{input}}  \n"
     "{identity}:  ").format

  def __init__(self, 
               role:CharacterRole, 
               model_name:str="gpt-3.5-turbo-16k"):
    self.role = role    
    self.llm = OpenAI(model_name=model_name,temperature=0.2)
    self.memory = ConversationBufferMemory(ai_prefix=self.identity,human_prefix="")
    prompt_template = CharacterPromptTemplate.format(role=str(self.role),identity=self.identity)
    self.prompt = PromptTemplate(input_variables=["history", "input"], template=prompt_template)
    self.chain = ConversationChain(llm=self.llm, prompt=self.prompt,memory=self.memory,verbose=False)

  @property
  def identity(self):
    return self.role.identity

  def __call__(self, input:str):
    return self.chain.run(input)