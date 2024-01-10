#Pokemon Showdown Voice Controller
#String Similarity Baseline Functions
#Aditya Pandey, Nitish Mallick, Savya Sachi Pandey, Vivek Kumar
#Note: Check the README and report for a full understanding of how this works

from poke_env import PlayerConfiguration
from poke_env.player import Player, RandomPlayer
from difflib import SequenceMatcher
import asyncio
import time
import speech_recognition as sr
import keyboard
from fuzzywuzzy import fuzz
import fuzzy
from transformers import BertModel
import torch
import torch.nn as nn
from transformers import BertTokenizer
import numpy as np
import pandas as pd
import openai
import time
import os
from functools import partial
import random
from collections import Counter
import tqdm
import re
import pyautogui


#################################################################	        

def similar(a, b):
    	return SequenceMatcher(None, a, b).ratio()


def mostsimilar(battle,input):
	#Return the index of most similar move

	moves=battle.available_moves
	switches=battle.available_switches

	max_score=0
	max_move=battle.available_moves[0]
	max_str=''

	for i in range(0,len(moves)):

		this_move=str(moves[i].id)

		score=similar(input,this_move)

		if score>max_score:
			max_score=score
			max_move=battle.available_moves[i]
			max_str=this_move


	print('Move chosen: ',max_str)
	return max_move