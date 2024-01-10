"""
https://python.langchain.com/en/latest/modules/agents/getting_started.html

"""
from dotenv import find_dotenv, load_dotenv

from langchain.llms import OpenAI
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

from langchain.agents.load_tools import get_all_tool_names


# Load environment variables
load_dotenv(find_dotenv())

print(get_all_tool_names())

llm = OpenAI()
tools = load_tools(["wikipedia", "llm-math"], llm=llm)

agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

result = agent.run(
    "What is the nelson pillar and calculate how many years ago it was destroyed if the current year is 2023"
)

print(result)


"""
> Entering new AgentExecutor chain...
 I need to look up the nelson pillar, then calculate how many years ago it was destroyed
Action: Wikipedia
Action Input: Nelson Pillar
Observation: Page: Nelson's Pillar
Summary: Nelson's Pillar (also known as the Nelson Pillar or simply the Pillar) was a large granite column capped by a statue of Horatio Nelson, built in the centre of what was then Sackville Street (later renamed O'Connell Street) in Dublin, Ireland. Completed in 1809 when Ireland was part of the United Kingdom, it survived until March 1966, when it was severely damaged by explosives planted by Irish republicans. Its remnants were later destroyed by the Irish Army.
The decision to build the monument was taken by Dublin Corporation in the euphoria following Nelson's victory at the Battle of Trafalgar in 1805. The original design by William Wilkins was greatly modified by Francis Johnston, on grounds of cost. The statue was sculpted by Thomas Kirk. From its opening on 29 October 1809 the Pillar was a popular tourist attraction, but provoked aesthetic and political controversy from the outset. A prominent city centre monument honouring an Englishman rankled as Irish nationalist sentiment grew, and throughout the 19th century there were calls for it to be removed, or replaced with a memorial to an Irish hero. 
It remained in the city as most of Ireland became the Irish Free State in 1922, and the Republic of Ireland in 1949. The chief legal barrier to its removal was the trust created at the Pillar's inception, the terms of which gave the trustees a duty in perpetuity to preserve the monument. Successive Irish governments failed to enact legislation overriding the trust. Although influential literary figures such as W. B. Yeats and Oliver St. John Gogarty defended the Pillar on historical and cultural grounds, pressure for its removal intensified in the years preceding the 50th anniversary of the Easter Rising, and its sudden demise was, on the whole, well-received by the public. Although it was widely believed that the action was the work of the Irish Republican Army (IRA), the police were unable to identify any of those responsible.
After years of debate and numerous proposals, the site was occupied in 2003 by the Spire of Dublin, a slim needle-like structure rising almost three times the height of the Pillar. In 2000, a former republican activist gave a radio interview in which he admitted planting the explosives in 1966, but, after questioning him, the Gardaí decided not to take action. Relics of the Pillar are found in Dublin museums and appear as decorative stonework elsewhere and its memory is preserved in numerous works of Irish literature.



Page: Horatio Nelson, 1st Viscount Nelson
Summary: Horatio Nelson, 1st Viscount Nelson, 1st Duke of Bronte  (29 September 1758 – 21 October 1805) was a British flag officer in the Royal Navy. His inspirational leadership, grasp of strategy, and unconventional tactics brought about a number of decisive British naval victories during the French Revolutionary and Napoleonic Wars. He is widely regarded as one of the greatest naval commanders in history.
Nelson was born into a moderately prosperous Norfolk family and joined the navy through the influence of his uncle, Maurice Suckling, a high-ranking naval officer. Nelson rose rapidly through the ranks and served with leading naval commanders of the period before obtaining his own command at the age of 20, in 1778. He developed a reputation for personal valour and a firm grasp of tactics, but suffered periods of illness and unemployment after the end of the American War of Independence. The outbreak of the French Revolutionary Wars allowed Nelson to return to service, where he was particularly active in the Mediterranean. He fought in several minor engagements off Toulon and was important in the capture of Corsica, where he was wounded and partially lost sight in one eye, and subsequent diplomatic duties with the Italian states. In 1797, he distinguished himself while in command of HMS Captain at the Battle of Cape St Vincent. Shortly after that battle, Nelson took part in the Battle of Santa Cruz de Tenerife, where the attack failed and he lost his right arm, forcing him to return to England to recuperate. The following year he won a decisive victory over the French at the Battle of the Nile and remained in the Mediterranean to support the Kingdom of Naples against a French invasion.
In 1801, Nelson was dispatched to the Baltic Sea and defeated neutral Denmark at the Battle of Copenhagen. He commanded the blockade of the French and Spanish fleets at Toulon and, after their escape, chased them to the West Indies and back but failed to bring them to battle. After a brief return to England, he took over the Cádiz blockade, in 1805. On 21 October 1805, the Franco-Spanish fleet came out of port, and Nelson's fleet engaged them at the Battle of Trafalgar. The battle became one of Britain's greatest naval victories, but Nelson, aboard HMS Victory, was fatally wounded by a French sharpshooter. His body was brought back to England, where he was accorded a state funeral.
Nelson's death at Trafalgar secured his position as one of Britain's most heroic figures. His signal just prior to the commencement of the battle, "England expects that every man will do his duty", is regularly quoted and paraphrased.  Numerous monuments, including Nelson's Column in Trafalgar Square, London, and the Nelson Monument in Edinburgh, have been created in his memory.

Page: Spire of Dublin
Summary: The Spire of Dublin, alternatively titled the Monument of Light (Irish: An Túr Solais), is a large, stainless steel, pin-like monument 120 metres (390 ft) in height, located on the site of the former Nelson's Pillar (and prior to that a statue of William Blakeney) on O'Connell Street, the main thoroughfare of Dublin, Ireland.


Thought: I now know the information I need to calculate the number of years ago the Nelson Pillar was destroyed
Action: Calculator
Action Input: 2023 - 1966
Observation: Answer: 57

Thought: I now know the final answer
Final Answer: The Nelson Pillar was destroyed 57 years ago in 1966.

> Finished chain.
The Nelson Pillar was destroyed 57 years ago in 1966.
"""