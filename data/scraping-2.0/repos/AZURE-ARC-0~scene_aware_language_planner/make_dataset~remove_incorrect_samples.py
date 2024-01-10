import openai
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_utils
# from evaluate import load
import pickle
import json
import re
import copy
from tqdm import tqdm
import os
import random
from pathlib import Path
import sys
import datetime
import pprint
import time
import multiprocessing

sys.path.append('../datasets/')

import add_preconds
import check_programs
from comm_unity import UnityCommunication
import utils_viz

GPU = 0
if torch.cuda.is_available():
	torch.cuda.set_device(GPU)
OPENAI_KEY = None  # replace this with your OpenAI API key, if you choose to use OpenAI API


# incorrect_samples = []
# incorrect_graphs = []
# incorrect_NL_aps = []
# incorrect_robot_aps = []
task_paths_all = list(Path('../datasets_old/init_graphs/').rglob("*.json"))
print(len(task_paths_all))

class Process(multiprocessing.Process):
	def __init__(self, graph_paths):
		super(Process, self).__init__()
		self.graph_paths = graph_paths

	def run(self):
		graph_paths = self.graph_paths
		for graph_path in tqdm(graph_paths):
			graph_path = str(graph_path)
			path = graph_path.replace('init_graphs', 'action_plans_robot').replace('json', 'txt')
			NL_path = graph_path.replace('init_graphs', 'action_plans_NL').replace('json', 'txt')

			robot_ap = open(str(path)).read().split('\n')
			init_graph = json.load(open(graph_path))

			try:
				preconds = add_preconds.get_preconds_script(robot_ap).printCondsJSON()
				info = check_programs.check_script(robot_ap, preconds, graph_path=None, inp_graph_dict=init_graph)
				message, final_state, graph_state_list, graph_dict, id_mapping, info, helper, modif_script = info
				success = (message == 'Script is executable')
			except:
				print("try fail")
				success = False

			if not success:
				print(graph_path)
				print(path)
				print(NL_path)

				assert os.path.exists(graph_path)
				assert os.path.exists(path)
				assert os.path.exists(NL_path)

				os.remove(graph_path)
				os.remove(path)
				os.remove(NL_path)

if __name__ == '__main__':
	procs = [None] * 13
	for i in range(13):
		print(i*500, (i+1)*500)
		procs[i] = Process(task_paths_all[i*500:(i+1)*500])

	for i in range(13):
		procs[i].start()

	for i in range(13):
		procs[i].join()
		print("joined", i)