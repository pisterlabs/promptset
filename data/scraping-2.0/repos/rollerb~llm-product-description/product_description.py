from langchain_core.runnables import Runnable
from langchain.llms.base import BaseLLM
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain.schema import StrOutputParser

def _get_product_description_prompt() -> ChatPromptTemplate:
  examples = [
    {
      "product": "PS3DQ0006",
      "description": """### PS3DQ0006: 3D 100 Degree Spray Angle Nozzle
* Inclined Fan Design: The 3D nozzle features an integrated fan incline and fan angle, maximizing efficacy and coverage for both contact and residual chemistries
* Drift Reduction: By tightening the spray pattern quality, the 3D nozzle can reduce drift potential compared to a conventional flat-fan nozzle
* SnapLock Quick-Change Assembly: The quick-change nozzle assembly includes a tip, cap, gasket, and integrated strainer for easy installation
* Enhanced Coverage: The droplet size and inclined pattern are designed to be installed alternating forwards and backwards on the boom, improving coverage on hard-to-hit targets like grass weeds and complex canopies
* PWM Compatibility: The 3D nozzle is compatible with ExactApply™ 30Hz high-frequency pulsing
"""
    },
    {
      "product": "RE330060",
      "description": """### RE330060: Rectangular Pedestal LED Work Light
* See the light of day, at night, with LED work lights from John Deere
* Proven to run cooler, last longer and cost less over the long haul than halogen and HID lights
* John Deere LED work lights boast some of the highest effective lumen outputs on the market
* Energy efficient results for longer battery & alternator life
"""
    }, 
    {
      "product": "LG265",
      "description": """### LG265: Home Maintenance Kit
* Makes it easy to maintain your John Deere lawn mower yourself with almost everything you need in one convenient box
* Includes high-quality OEM John Deere parts.
* Ensures optimal performance because the maintenance kits is designed specifically for John Deere mowers
* Ensures optimal performance because the maintenance kits is designed specifically for John Deere mowers
* Kit Includes: AM116304 - Inline Fuel Filter, AM125424 - Engine Oil Filter, DKE984696 - Oil Change Label, 2 X M805853 - Spark Plug, MIU12554 - Primary Air Filter Element, MIU12555 - Secondary Air Filter Element, 2 X TY22029 - 4-Cycle Engine Oil, TURF-GARD™, 10W-30, API SN/GF6, 946 ml (32 Fluid Oz)""" 
    }
  ]
  example_prompt = ChatPromptTemplate.from_messages([
    ("human", "Product: {product}\nDescription:"),
    ("ai", "{description}")
  ])
  few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt = example_prompt,
    examples=examples
  )
  chat_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at providing product descriptions. You will generate a description for the <product> using the examples as a guide and the <context> provided. You will respond in a helpful, correct, and professional manner. Always uses line breaks and markdown to increase readability. Utilize bullet points where it makes sense. DO NOT include details about return policies, warranty information, or price. Make the description include no more than 5 bullet points.
     """),
    ("human", "<context>{context}</context>"),
    few_shot_prompt,
    ("human", """Product: {product}\nDescription:""")
  ])
  return chat_prompt

def create_product_description_chain(llm: BaseLLM) -> Runnable:
  description_prompt = _get_product_description_prompt()
  chain = description_prompt | llm | StrOutputParser()  
  return chain