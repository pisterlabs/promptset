import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import openai, os, ast
from scipy.spatial.distance import euclidean
from tqdm import tqdm
from utils import ada_embeddings
import seaborn as sns
import statsmodels.api as sm
import argparse

#Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--only_stem', default=False, help='This experiment only makes sense if LLM was ran on the entire MMLU dataset')
parser.add_argument('--engine', default="text-davinci-002", help='engine is the name of the LLM used for knowing which folder to look in')
args = parser.parse_args()

if args.only_stem:
    raise NotImplementedError("This experiment only makes sense if LLM was ran on the entire MMLU dataset!")

ks = [0, 1, 2] #number of previous points
ns = [0, 2] #number of previous tasks. Essentially 0 is in-context learning and 2 is meta-in-context learning
task_sims = np.zeros((56, len(ks)))
task_accs = np.zeros((56, len(ks)))

# Loop over all test points
for idxk, k in tqdm(enumerate(ks)):
    # Loop over all previous tasks (0 is in-context learning and 2 is meta-in-context learning)
    for idxn, n in enumerate(ns):
        meta_icl_results = test_list = glob.glob(f"results/{args.engine}/meta_in_context_k=" + str(k) + "_n=" + str(n) + "*.csv")
        meta_icl_acc = []
        # Loop over all test points
        for i, meta_icl_result in enumerate(meta_icl_results):
            df = pd.read_csv(meta_icl_result)
            meta_icl_acc.append((df['gt'] == df['choice']).mean())
            if idxn == 1: 
                # This means that we are in the meta-in-context learning case
                current_task = df['task'][0].split('/')[-1].replace('_test.csv', '')
                previous_tasks_rows = df['meta_tasks'].unique()
                task_accs[i, idxk] = meta_icl_acc[-1]
                #L2 norm
                for previous_tasks in previous_tasks_rows:
                    #TODO: Change below line which is super slow.
                    task_sims[i, idxk] += np.mean([np.linalg.norm(ada_embeddings(current_task, args.engine) - ada_embeddings(previous_task, args.engine)) for previous_task in ast.literal_eval(previous_tasks.replace("' '", "','"))])

#plot regression onto accuracy barplot with regressors trial and task similarity

# Create dataframe for processing the data for regression
df = pd.DataFrame()
df['task_similarity'] = - task_sims.flatten() # negative because we want similarity and not distance
df['trial'] = [i for i in range(1, 4) for _ in range(56)]
df['accuracy'] = task_accs.flatten()
#Standardize
df['task_similarity'] = (df['task_similarity'] - df['task_similarity'].mean()) / df['task_similarity'].std()
df['trial'] = (df['trial'] - df['trial'].mean()) / df['trial'].std()
df['accuracy'] = (df['accuracy'] - df['accuracy'].mean()) / df['accuracy'].std()

# Regression
X = df[['task_similarity', 'trial']]
y = df['accuracy']
model = sm.OLS(y, X).fit()
print(model.summary())

# barPlot of task similarity and trial onto accuracy
plt.bar(x=model.params.index, height=model.params.values, yerr=model.bse.values)
plt.savefig(f'plots/Barplot_{args.engine}{"_stem" if args.only_stem else ""}.png')

