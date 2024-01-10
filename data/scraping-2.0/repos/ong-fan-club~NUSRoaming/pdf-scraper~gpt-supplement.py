import glob
import os
import re

import fitz
import openai
import ujson

old = """University of Hohenheim (Hohenheim)
King's College London
Waseda University
Queen's University at Kingston
Tongji University
Mahidol University
National Yang Ming Chiao Tung University
Peking University
National Taiwan University
University of Virginia
University of Ulm
University of Adelaide
Emory University
University of North Carolina at Chapel Hill
University of Southern California
The College of William and Mary
University of Connecticut
Purdue University
Monash University
Cornell University - College of Human Ecology (CHE)
Australian National University
University of Notre Dame
Norwegian University of Science and Technology (NTNU)
University of Western Australia
Arizona State University
University of Lausanne
University of Concordia
Kyoto University
University of Amsterdam, The Netherlands
Tecnológico de Monterrey
University of Florida
École Nationale Supérieure des Mines de Paris (MINES ParisTech)
Communauté Université Grenoble Alpes
Utrecht University
Georgia Institute of Technology
The Chinese University of Hong Kong
Tallinn University of Technology (TalTech)
University of Zurich
The University of Sheffield
Bilkent University
The University of Texas at Austin
University of Glasgow
Osaka University
University of Gothenburg
University of St Andrews
Boston College
Carnegie Mellon University
Cornell University - College of Agriculture & Life Sciences (CALS)
Delft University of Technology
University of Auckland
RWTH Aachen University
The University of Warwick
Korea Advanced Institute of Science and Technology (KAIST)
Uppsala University
Technical University of Darmstadt (TUD)
University of Wisconsin Madison
City University of Hong Kong
Pohang University of Science and Technology (POSTECH)
Keio University
University of Birmingham
Lund University
Warsaw University of Technology (WUT)
University of Mannheim
The Hong Kong Polytechnic University
Dalhousie University
University of Arizona
Clarkson University
University of Canterbury
Vilnius University
Technical University of Munich
University of Maryland
University of Calgary
University of Guelph
AARHUS UNIVERSITY
University of Tokyo
Korea University
Technical University of Denmark
McGill University
University of Copenhagen
National Cheng Kung University
Budapest University of Technology and Economics
Imperial College London
University of Konstanz
University of Tuebingen (Eberhard Karls Universität Tübingen)
Kyushu University
Boston University
University of Zagreb
University of Pittsburgh
University of Bristol
University of Oslo
University of Ottawa
Chulalongkorn University
"""
unis = """University of Miami
University College London
École Polytechnique Fédérale de Lausanne (EPFL)
Ludwig-Maximilians-University of Munich (LMU)
University of Stuttgart
Rice University
Newcastle University
Yonsei University
Simon Fraser University
Shanghai Jiao Tong University
Université de Technologie Compiègne (UTC)
Durham University
Pennsylvania State University
Brandeis University
Texas A&M University
The Hebrew University of Jerusalem
CentraleSupélec
Tsinghua University
University of Victoria
University of Washington, Seattle
Cracow University of Technology
University of Waterloo
University of Innsbruck
University of Melbourne
Humboldt University of Berlin (HUB)
University of Leeds
Western University
Case Western Reserve University
KTH Royal Institute of Technology
University of Hawaiʻi at Mānoa
University of Bologna
University of Queensland
The University of Hong Kong
Universidad Autónoma de Madrid
Georgetown University
The University of Bath (UoB)
The George Washington University
University of British Colombia
Xiamen University
Tokyo Institute of Technology
Albert-Ludwig University of Freiburg (University of Freiburg)
Karlsruhe Institute of Technology (KIT)
University of New South Wales
ETH Zurich (Swiss Federal Institute of Technology)
Stockholm University
University College Cork
Ruprecht Charles University of Heidelberg
University College Dublin (UCD)
Jagiellonian University
University of Pretoria
Nanjing University
Chongqing University
University of Toronto
Tohoku University
University of Geneva
University of Colorado Boulder
University of California
University of Alaska Fairbanks
Rheinische Friedrich-Wilhelms-Universität Bonn (University of Bonn)
Eindhoven University of Technology
Chalmers University of Technology
Carleton College
Seoul National University
Sciences Po Paris
The University of Nottingham
Reichman University
Telecom ParisTech
Hamburg University of Technology
Université de Technologie de Troyes (UTT)
University of Manchester
University of Auckland
University of Pennsylvania
University of Michigan at Ann Arbor
Victoria University of Wellington
York University
Sabanci University
Nagoya University
The University of Edinburgh
The University of York
Institut National des Sciences Appliquées de Lyon (INSA Lyon)
National Tsing Hua University
Fudan University
Trinity College Dublin
Technion – Israel Institute of Technology
University of Oregon
Free University Berlin (Freie Universität Berlin - FUB)
Zhejiang University
Northwestern University
Alberta University
The University of Liverpool (UoL)
University of Sydney"""

# load openai api key from env
openai.api_key = os.environ.get("OPENAI_API_KEY")
prompt = """I want to curate information about partner universities that are available to National University of Singapore students to go on exchange with.

I want you to produce a combined final JSON file with the key information requested. The JSON must be valid JSON. Output ONLY the valid JSON and nothing more.

I am going to give you a university_name. Using any information you have from the internet or prior knowledge, tell me information about the university that map with these keys:
- university_name (provided name)
- gpt_university_description (text describing the university)
- gpt_university_address (the address of the university)
- gpt_university_city (the city and/or state the university is located in)
- gpt_nearest_airport (what is the nearest international airport to the university?)
- gpt_location_cost_of_living (text on cost of living in the city the university is in)
- gpt_location_weather (text describing the weather of the city)
- gpt_location_description (text describing the city in between 50 to 100 words)
- gpt_location_crime (text describing crime rate/situation of the city)
- gpt_location_transportation (text describing public transportation options in the city)
- gpt_location_halal (availability of halal food options in city. Output a score between 1 to 3, where 1 means extremely limited or no vegetarian options, and 3 means an abundance of vegetarian options)
- gpt_location_vegetarian (availability of vegetarian food options in the city. Output a score between 1 to 3, where 1 means extremely limited or no vegetarian options, and 3 means an abundance of vegetarian options)

UNIVERSITY NAME:
"""


ctr = 92
s = set()
for uni in unis.split("\n"):
    print(uni)
    response = openai.ChatCompletion.create(
        model="gpt-4", messages=[{"role": "user", "content": prompt + uni}]
    )
    print(response)
    res = response.choices[0].message.content
    print(res)
    # save res to a text file
    with open(f"{ctr}.txt", "w") as f:
        f.write(res)
    ctr += 1
    # break
print(s)
