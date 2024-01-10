# Author: Scott Spillias
# Email: scott.spillias@csiro.au

## Import Packages
import os
import sys
import xlrd
import pandas as pd 
import openai
pd.options.mode.chained_assignment = None  # default='warn'
import re
import numpy as np
import time
import random
import string

ScreeningCriteria = [
   "Is it possible that the Title and Abstract summarise a paper with a case study from: Cook Islands, Federated States of Micronesia, Fiji, Kiribati, Marshall Islands, Nauru, Niue, Palau, Papua New Guinea, Samoa, Solomon Islands, Tonga, Tuvalu, Vanuatu",
   "Is it likely that the Title and Abstract summarise a paper that will discuss fisheries and/or marine resource management?",
   "Is it likely that the Title and Abstract summarise a paper that will discuss a community-based approach? Exclude if there is a focus on co-management.",
 #  "Is it likely that the Title and Abstract summarise a paper that is part of the peer reviewed published literature? Exclude grey literature, conference reports, and book chapters."
    ]

#proj_location = sys.path[0] # Directory name of project.

## Set meta-parameters
debug = False # Run fewer articles for debugging purposes

n_reviewers = 5 # Number of AI agents to independently ask

skip_criteria = False # Will skip remaining screening criteria if one fails across all AI agents

screen_name = 'pilot' # Name of spreadsheet with Abstract and Title *without extensions

## Set-up Screening Run
excel_sheet = '1_' + screen_name + '.xls'

openAI_key = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" # Insert your OpenAI API key

model_to_use = "gpt-3.5-turbo-0301" # Currently only set-up for "gpt-3.5-turbo-0301"

temperature = 0 # increasing this number will lead to less 'truthful' responses

n_retries = 10 # Number of times to re-query OpenAI in event of disconnection

save_frequency = 10 # How often to save intermediate results.

for rand_seed in [False]: # Insert random string into prompts for uniqueness
   print(rand_seed)

   note = 'rand-' + str(rand_seed) # This will append a note to the output files

   exec(open('../Code/1_Screening.py').read())