#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
[env]
# Conda Environment
# NO CONDA ENV

conda create --name ai_chatgpt_prompts python=3.9.13
conda info --envs
source activate ai_chatgpt_prompts
conda deactivate
# if needed to remove
conda env remove -n [NAME_OF_THE_CONDA_ENVIRONMENT]


# update conda 
conda update -n base -c defaults conda

[filename]
016_project_1_python_documentation_extract_keywords_chatgpt_api.py

[path]
cd /Users/brunoflaven/Documents/03_git/BlogArticlesExamples/ai_chatgpt_prompts/project_1_python_documentation_chatgpt_api/



[run]
python 016_project_1_python_documentation_extract_keywords_chatgpt_api.py

pip install Random

+ GREAT EXAMPLE FROM OPENAI.COM
https://platform.openai.com/examples


Extract keywords from a block of text. At a lower temperature it picks keywords from the text. At a higher temperature it will generate related keywords which can be helpful for creating search indexes.





"""

import os
import openai

# personal configuration
import config_values.values_conf as conf

OPENAI_ORGANIZATION = conf.OPENAI_ORGANIZATION
OPENAI_API_KEY = conf.OPENAI_API_KEY

# quick and dirty
openai.organization = OPENAI_ORGANIZATION
# PAID ONE DO NOT DISPLAY
openai.api_key = OPENAI_API_KEY

response = openai.Completion.create(
    model="text-davinci-003",
    
    # prompt="Extract keywords from this text:\n\nBlack-on-black ware is a 20th- and 21st-century pottery tradition developed by the Puebloan Native American ceramic artists in Northern New Mexico. Traditional reduction-fired blackware has been made for centuries by pueblo artists. Black-on-black ware of the past century is produced with a smooth surface, with the designs applied through selective burnishing or the application of refractory slip. Another style involves carving or incising designs and selectively polishing the raised areas. For generations several families from Kha'po Owingeh and P'ohwhóge Owingeh pueblos have been making black-on-black ware with the techniques passed down from matriarch potters. Artists from other pueblos have also produced black-on-black ware. Several contemporary artists have created works honoring the pottery of their ancestors.",
    
    prompt="Extract keywords from this text:\n\nWhat does the future hold for Wagner in Africa after the failed rebellion?\nRussian Foreign Minister Sergei Lavrov on Monday said that Wagner group troops would continue to operate in Mali and the Central African Republic following the rebellion led by their commander Yevgeny Prigozhin last weekend. Lavrov’s words came amid questions over the private militia’s role in Africa after more than five years of deployment to the continent.\nMembers of the Close Protection Unit for Central African Republic President Touadera, composed of Russian private security company operatives from Sewa Security, are seen in Berengo on August 4, 2018.\nMembers of the Close Protection Unit for Central African Republic President Touadera, composed of Russian private security company operatives from Sewa Security, are seen in Berengo on August 4, 2018. © Florent Vergnes, AFP\nThe Wagner group’s mutiny against Moscow last weekend has raised questions over the private militia’s presence in Africa.\nFor more than five years, thousands of Wagner group troops have deployed to Africa. In 2018 they were sent to support Central African Republic President Faustin-Archange Touadera as he grappled with an armed rebellion. In December 2021 they deployed to Mali in the wake of a coup and they have operated in Sudan since the end of 2017. Wagner troops have also been spotted in Libya and Mozambique.\nBut Lavrov vowed on Monday that the “events” of the last weekend would not impact the militia’s operations on the continent.\nThe Russian \"instructors\" and “private military contractors” in both Mali and the Central African Republic will continue their work, Lavrov said in an interview with Russia Today.\nPrigozhin’s rebellion will not change anything in Russia's ties with its allies, Lavrov added. \"There have been many calls(from foreign partners) to President(Vladimir) Putin ... to express their support, \" he said.\nNo African government has officially commented on the events of the past weekend. But according to Cyril Payen, senior reporter for FRANCE 24, Russia \"has undoubtedly become a slightly less reliable partner\" since Prigozhin’s rebellion.\"You can bet that people in Bangui and Bamako are wondering what the future holds, \" Payen added.\"The Malian state is now engaged in a double partnership, with the Russian state – the Putin camp – and with the Wagner group – the Prigozhin camp. Until now, this did not make much difference, but that could change if the two camps don’t reconcile in the long term, \" said lawyer and political scientist Oumar Berté in an interview with FRANCE 24’s sister radio station Radio France Internationale.\nOverlapping interests\nHowever, a senior official in the Central African Republic’s presidency told AFP that Russia will continue to operate in the central African country, with or without Wagner.\"The Central African Republic signed (in 2018, editor's note) a defence agreement with the Russian Federation, not with Wagner, \" said Fidele Gouandjika, special adviser to President Touadera.\"Russia has subcontracted with Wagner. If Russia no longer agrees with Wagner, then it will send us a new contingent.\"\n\nA bridgehead for Russian ambitions on the continent, the Central African Republic is particularly dependent on the Russian militia, whose men even work as private protection officers for Touadera.\n\nSome 1, 500 Wagner troops have been deployed to Mali since 2021. The paramilitary group has developed close ties with the junta in power, helping to train soldiers as well as taking part in operations to combat terrorist groups.\n\nPrigozhin’s men have also been seen in Libya, Sudan and Mozambique. Since the Wagner's group arrival in Africa, the UN, international NGOs and French authorities have regularly accused the paramilitary group of committing abuse and crimes against civilians.\n\nWagner always uses the same strategy every time it advances: disinformation campaigns(based on rejecting former colonial powers) and an offer of security in exchange for the exploitation of natural resources to supply Prigozhin’s war chest and serve the Kremlin’s interests.\n\nIn Sudan, the partnership between Wagner and the Rapid Support Forces(RSF), led by the junta’s number two, General Mohammed Hamdan Daglo, has enabled the paramilitary group to profit from illegal gold trafficking. It has also enabled them to organise the transport of the metal straight into the coffers of the Russian state, helping to swell its gold reserves and circumvent Western sanctions.",
    temperature=0.5,
    max_tokens=60,
    top_p=1.0,
    frequency_penalty=0.8,
    presence_penalty=0.0
)

print('\n --- RESULT')
print(response)
# print(response.choices)
# desired_text = response.choices[0].text
# # print(desired_text)


