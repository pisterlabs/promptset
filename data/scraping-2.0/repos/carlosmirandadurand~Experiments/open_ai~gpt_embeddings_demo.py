#%%##################################################################################
# OpenAI - Embeddings Demo
# Original source code: 
#       https://platform.openai.com/docs/guides/embeddings/use-cases
#       https://github.com/openai/openai-cookbook/blob/fde2a6474db03bf61d8def356fead62784fe45e9/examples/Get_embeddings_from_dataset.ipynb
#####################################################################################


import os
import pandas as pd
import openai
import tiktoken

from openai.embeddings_utils import get_embedding
from dotenv import load_dotenv


#%% 
# Connect to Open AI

load_dotenv()
openai.organization = os.getenv('openai_organization_id')
openai.api_key      = os.getenv('openai_organization_key')


#%% 
# Generate embeddings

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']



#%% 
# Test

text_items = [

"""On 26 July 2023, a coup d'état occurred in Niger, during which the country’s presidential guard removed and detained President Mohamed Bazoum. Subsequently, General Abdourahamane Tchiani, the Commander of the Presidential Guard, declared himself the leader of a military junta, established the National Council for the Safeguard of the Homeland, after confirming the success of the coup.[3][4][5][6].
In response to this development, the Economic Community of West African States (ECOWAS) issued an ultimatum on 30 July, giving the coup leaders in Niger one week to reinstate Bazoum, with the threat of international sanctions and potential use of force.[7][8] When the deadline of the ultimatum expired on 6 August, no military intervention was initiated; however, on 10 August, ECOWAS took the step of activating its standby force.[9][10][11][12] ECOWAS previously intervened in the Gambia to restore democracy following the country's 2016–2017 constitutional crisis.
All member states of ECOWAS that are actively participating, except for Cape Verde, have pledged to engage their armed forces in an ECOWAS-led military intervention against the Nigerien junta,[13] should such an intervention be launched. Conversely, the military juntas in Burkina Faso and Mali announced they would send troops in support of the junta if such military intervention were to ever be launched.[14][15]
""",

"""Niger or the Niger[13][14] (/niːˈʒɛər, ˈnaɪdʒər/ nee-ZHAIR, NY-jər,[15][16] French: [niʒɛʁ]),[a] officially the Republic of the Niger[13][14] (French: République du Niger; Hausa: Jamhuriyar Nijar), is a landlocked country in West Africa. It is a unitary state bordered by Libya to the northeast, Chad to the east, Nigeria to the south, Benin and Burkina Faso to the southwest, Mali to the west, and Algeria to the northwest. It covers a land area of almost 1,270,000 km2 (490,000 sq mi), making it the largest landlocked country in West Africa. Over 80% of its land area lies in the Sahara. Its predominantly Muslim population of about 25 million[17][18] live mostly in clusters in the south and west of the country. The capital Niamey is located in Niger's southwest corner.
According to Multidimensional poverty index (MPI) report of 2023, Niger is one of the poorest countries in the world.[19] Some non-desert portions of the country undergo periodic drought and desertification. The economy is concentrated around subsistence agriculture, with some export agriculture in the less arid south, and export of raw materials, including uranium ore. It faces challenges to development due to its landlocked position, desert terrain, low literacy rate, jihadist insurgencies, and the world's highest fertility rates due to birth control not being used and the resulting rapid population growth.[20]
Its society reflects a diversity drawn from the independent histories of some ethnic groups and regions and their period living in a single state. Historically, Niger has been on the fringes of some states. Since independence, Nigeriens have lived under five constitutions and three periods of military rule. After the military coup in 2010 up until 2023, Niger became a multi-party state. A majority of the population lives in rural areas.
""",

"""A wooden roller coaster is a type of roller coaster classified by its wooden track, which consists of running rails made of flat steel strips mounted on laminated wood. The support structure is also typically made of wood, but may also be made of steel lattice or truss, which has no bearing on a wooden coaster's classification. The type of wood often selected in the construction of wooden coasters worldwide is southern yellow pine, which grows abundantly in the southern United States, due to its density and adherence to different forms of pressure treatment.
Early wooden roller coaster design of the 19th century featured a single set of wheels running on top of the track, which was common in scenic railway rides. John A. Miller introduced side friction coasters and later underfriction coasters in the early 20th century, which added additional sets of wheels running along multiple sides of the track to allow for more intense ride design with sharper turns and steeper drops. The underfriction design became commonplace and continues to be used in modern roller coaster design.
Traditionally, wooden roller coasters were not capable of featuring extreme elements such as inversions, near-vertical drops, and overbanked turns commonly found on steel roller coasters after the introduction of tubular steel track by Arrow Development in 1959. Son of Beast at Kings Island made history in 2000 by incorporating the first successful attempt of an inversion on a wooden coaster, a vertical loop made of steel. A decade later, the introduction of Topper Track by Rocky Mountain Construction allowed for new possibilities, with corkscrews, overbanked turns, and other inverting elements appearing on wooden coasters such as Outlaw Run at Silver Dollar City and Goliath at Six Flags Great America.
""",

"""Africa is the world's second-largest and second-most populous continent, after Asia in both aspects. At about 30.3 million km2 (11.7 million square miles) including adjacent islands, it covers 20% of Earth's land area and 6% of its total surface area.[7] With 1.4 billion people[1][2] as of 2021, it accounts for about 18% of the world's human population. Africa's population is the youngest amongst all the continents;[8][9] the median age in 2012 was 19.7, when the worldwide median age was 30.4.[10] Despite a wide range of natural resources, Africa is the least wealthy continent per capita and second-least wealthy by total wealth, ahead of Oceania. Scholars have attributed this to different factors including geography, climate, tribalism,[11] colonialism, the Cold War,[12][13] neocolonialism, lack of democracy, and corruption.[11] Despite this low concentration of wealth, recent economic expansion and the large and young population make Africa an important economic market in the broader global context.
"""
]

#%%

get_embedding(text_items[0])


#%% END

