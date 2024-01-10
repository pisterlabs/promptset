import openai
import time

import pandas as pd

from GNED.config import *

openai.api_key = os.environ["OPENAI_API_KEY"]
contexts = [
    'Eco Market is a retail chain of supermarkets based in Tirana Albania. Established in February 2013 the chain has expanded its operations with the opening of 19 stores mainly in the TiranDurrs metropolitan region and in Fier region. It manages a wide variety of products with the Eco brand name which it supplies to other major distribution chains in the country such as <NE>Big Market</NE> Conad and Carrefour.',
    'The third group contains geophytic peperomias. These plants have leaves that fall off in the colder dry season survive due to their underground tubers and grow the leaves back as more rain falls. Examples include "P. macrorhiza" "<NE>P. peruviana</NE>" and "P. umbilicata". Currently just under 50 species of geophytic peperomias are known but new ones continue to be discovered.',
    'The castle was built between 1290 and 1304 by Hughes II de Bouville and <NE>Hugues III</NE> Lords of Farcheville and Bouville. The Chteau de Farcheville is a rare example of rural openfilled castle of the medieval period. The great hall was built in 1291 and the castle chapel was consecrated in 1304. Both father and son were chamberlain to Philip IV of France. The structure possesses a rare northern French example of arched machicolations on buttresses more characteristic of military architecture in the Languedoc. The castle passed to the family of Chtillon in the fifteenth century.',
    'In summer 2010 "Helsingin Sanomat" evaluated the products offered by the Pappagallo kiosk in the Merituuli shopping centre in <NE>Suomenoja</NE> Espoo. According to the reporter the raspberry sorbet was "like eating raspberries it even had the seeds". The reporter also described the chocolate ice cream as dark and strong the liquorice ice cream as tasting like salty liquorice. The vegan board of Animalia chose Pappagallo\'s mango sorbet as the milkfree ice cream product of the year in 2011 describing it as fresh with an intense mango flavour fullbodied juicy tasting like a real mango fruity and suitably sweet.',
    'Geet or The Song is a 1944 Bollywood film. "Geet" was directed by S. U. Sunny and produced by Abdul Rashid Kardar. The film starred <NE>Shahu Modak</NE> Nirmala Devi Aamir Ali Shakir Ali and Chandabai. The music for the film was composed by Naushad with lyrics by D. N. Madhok.',
    'Gastropoda some airbreathing land snails including species in the genera "Helix <NE>Cernuella</NE> Theba Helicella Achatina" and "Otala" commonly aestivate during periods of heat. Some species move into shaded vegetation or rubble. Others climb up tall plants including crop species as well as bushes and trees and will also climb manmade structures such as posts fences etc.',
    'Lago Verde is a Chilean commune located at the headwaters of the <NE>Cisnes River</NE> in Coyhaique Province Aisn Region. The commune is administered by the municipality of Lago Verde the principal settlement.',
    'Silakheri railway station is an Indian railway station on the <NE>IndoreGwalior line</NE> under the Ratlam railway division of Western Railway zone. This is situated beside National Highway 3 at Chhota Mahalasapura in Dewas district of the Indian state of Madhya Pradesh.',
    'The 1963 Japan Series was the last held at Heiwadai. In 1969 a gamefixing and gambling conspiracy dubbed the Black Mist Scandal was uncovered and resulted in several key Lions players being suspended or banned from baseball. The loss of these players dramatically weakened Nishitetsu and the team finished in last place for three consecutive seasons from 1970 to <NE>1972</NE> the team would not recover from the scandal for the rest of the 1970s. By the end of 1972 the average attendance at Heiwadai had dropped to 4900people per game or 320000 annually. In the offseason Nishitetsu sold the team to former Lotte Orions owner Nagayoshi Nakamura who renamed the team the Taiheiyo Club Lions after securing a sponsorship from golf company Taiheiyo Club.',
    'As is the case wherever the US military is stationed there is an American Forces Network (AFN) station. It transmits on FM on 102.3MHz from Fernmeldeturm Frauenkopf and on AM on 1143kHz from <NE>Hirschlanden transmitter</NE>.'
]
descriptions = [
    'Albanian supermarket chain',
    'species of plant',
    'chamberlain of Philip IV of France',
    'neighborhood in Espoo Finland',
    'Indian actor',
    'genus of molluscs',
    'river in Chile',
    'railway line in India',
    'sports season',
    'architectural structure'
]


def get_response(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


def send_requests(data_path, start_index, end_index):

    df = pd.read_csv(data_path, delimiter='\1')
    df = df.iloc[start_index:end_index]

    with open('gpt-response.csv', 'a+') as f:
        for context, desc in zip(df['contexts'].values, df['entity_description'].values):
            try:
                messages = [
                    {'role': 'system', 'content': """
                    Your task is to take a paragraph as a context in the input, and generate the Wikidata description of the
                    entity that is tagged by the <NE> and </NE> within that paragraph. Each description should be a single 
                    sentence and should not exceed 100 tokens. 
                    """},
                    {'role': 'system', 'name': 'example_user', 'content': contexts[0]},
                    {'role': 'system', 'name': 'example_assistant', 'content': descriptions[0]},
                    {'role': 'system', 'name': 'example_user', 'content': contexts[1]},
                    {'role': 'system', 'name': 'example_assistant', 'content': descriptions[1]},
                    {'role': 'system', 'name': 'example_user', 'content': contexts[2]},
                    {'role': 'system', 'name': 'example_assistant', 'content': descriptions[2]},
                    {'role': 'system', 'name': 'example_user', 'content': contexts[3]},
                    {'role': 'system', 'name': 'example_assistant', 'content': descriptions[3]},
                    {'role': 'system', 'name': 'example_user', 'content': contexts[4]},
                    {'role': 'system', 'name': 'example_assistant', 'content': descriptions[4]},
                    {'role': 'user', 'content': context}
                ]
                response = get_response(messages)
                print(desc)
                print(response)
                f.write(f'{context}\1{desc}\1{response}\n')
                time.sleep(20)
            except:
                print('not ready')
                time.sleep(30)
