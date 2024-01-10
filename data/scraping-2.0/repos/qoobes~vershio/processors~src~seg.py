import json
import os

import openai
from dotenv import load_dotenv

from gpt import GPT, Example


def segment(prompt):
    openai.api_key = os.environ.get("GPT_KEY")

    gpt = GPT(temperature=0, max_tokens=500, append_output_prefix_to_query=False, output_prefix="")

    gpt.add_example(Example("""George Washington (February 22, 1732[b] – December 14, 1799) was an American political leader, military general, statesman, and Founding Father who served as the first president of the United States from 1789 to 1797. Previously, he led Patriot forces to victory in the nation's War for Independence. He presided at the Constitutional Convention of 1787, which established the U.S. Constitution and a federal government. Washington has been called the "Father of His Country" for his manifold leadership in the formative days of the new nation.

Washington's first public office was serving as official Surveyor of Culpeper County, Virginia from 1749 to 1750. Subsequently, he received his initial military training and a command with the Virginia Regiment during the French and Indian War. He was later elected to the Virginia House of Burgesses and was named a delegate to the Continental Congress, where he was appointed Commanding General of the Continental Army. He commanded American forces, allied with France, in the defeat and surrender of the British during the Siege of Yorktown. He resigned his commission after the Treaty of Paris in 1783.

Washington played a key role in adopting and ratifying the Constitution and was then twice elected president by the Electoral College. He implemented a strong, well-financed national government while remaining impartial in a fierce rivalry between cabinet members Thomas Jefferson and Alexander Hamilton. During the French Revolution, he proclaimed a policy of neutrality while sanctioning the Jay Treaty. He set enduring precedents for the office of president, including the title "Mr. President", and his Farewell Address is widely regarded as a pre-eminent statement on republicanism.
""", "[\"George Washington was born on February 22, 1732.\", \" George Washington died on December 14, 1799\", \"George Washington was an American political leader and military general, statesman.\", \"George Wahington was the Founding Father who served as the first president of the United States from 1789 to 1797.\", \"George Washington led the Patriot forces to victory in the nation's War for Independence. \", \"George Washington presided at the Constitutional Convention of 1787, which established the U.S. Constitution and a federal government.\", \"George Washington has been called the 'Father of His Country' for his manifold leadership in the formative days of the new nation.\", \"George Washington was serving as official Surveyor of Culpeper County, Virginia from 1749 to 1750. \", \"George Washington was later elected to the Virginia House of Burgesses and was named a delegate to the Continental Congress.\", \"George Washington was appointed Commanding General of the Continental Army.\", \"George Washington commanded American forces, allied with France, in the defeat and surrender of the British during the Siege of Yorktown. \", \"George Washington resigned his commission after the Treaty of Paris in 1783.\", \"George Washington played a key role in adopting and ratifying the Constitution \", \"George Washington was then twice elected president by the Electoral College.\", \"George Washington during the French Revolution, proclaimed a policy of neutrality while sanctioning the Jay Treaty. \"]"))

    sjExample = Example("""Apple was founded in April 1976 by the late Steve Jobs and Ronald Wayne. Wayne would leave the Apple company only three months after its founding to take a job with Atari, and Jobs would end up buying the company from him.""",
    "[\"Apple was founded in April 1976 by the late Steve Jobs and Ronald Wayne.\", \"Ronald Wayne would leave the Apple company only three months after its founding to take a job with Atari, and Jobs would end up buying the company from him.\"]"
    )
    gpt.add_example(sjExample)


    gpt.add_example(Example("""Adenosine triphosphate (ATP) is an organic compound and hydrotrope that provides energy to drive many processes in living cells, such as muscle contraction, nerve impulse propagation, condensate dissolution, and chemical synthesis. Found in all known forms of life, ATP is often referred to as the "molecular unit of currency" of intracellular energy transfer. [2] When consumed in metabolic processes such as cellular respiration, it converts either to adenosine diphosphate (ADP) or to adenosine monophosphate (AMP). Other processes regenerate ATP so that the human body recycles its own body weight equivalent in ATP each day.[3] It is also a precursor to DNA and RNA, and is used as a coenzyme.""", "[\"ATP is an organic compound and hydrotrope.\", \"ATP provides energy to drive many processes in living cells.\", \"ATP is often referred to as the 'molecular unit of currency' of intracellular energy transfer.\"]")) 


    out = gpt.submit_request(prompt).choices[0].text
    return out

out = segment("Cooking and eating will likely look similar in the first half of 2021 in the U.S., even as the Covid-19 vaccine rolls out, the spread slows and restrictions gradually lift, according to the market research firm. In the second half of the year, forecasts are more mixed. IRI expects grocery spending to drop and dining out to bounce back to near pre-pandemic levels. The average household will spend about half of their dining dollars away from home, according to IRI. It dipped to nearly 30% at the height of the global health crisis. “The curiosity with ethnic flavors and things like that, that’s not going to slow down,” he said. “People have been introduced to them. They like them.” That heightened interest in more adventurous flavors has lifted sales for other companies, too. Ethnic brands, such as Hispanic brand Goya Foods, have attracted new and repeat customers. Spice company McCormick acquired hot-sauce maker Cholula in November to cash in on demand for spicy sauces. And this summer, PepsiCo’s Frito-Lay division decided to sell top flavors from around the globe in potato chip form in the U.S. — including Brazilian Picanha and Chinese Szechuan Chicken.")
print(out)
