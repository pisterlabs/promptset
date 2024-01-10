import openai
import jsonlines
import time
import pandas as pd
import numpy as np
import random
import json
import sys
import os
from rdkit.Chem import rdMolDescriptors as rdmd
# Set up your API key and model parameters
#model_engine = 'gpt-3.5-turbo' # You can choose a different model if desired
model_engine = 'text-davinci-003'
temperature = 0.3 
num_generations = 10
import openai
import jsonlines
import time
import subprocess
import re
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import RDConfig
from rdkit.Chem import QED
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer


directory_path = './LMLF/'
target_file_path = './LMLF/one-box/drd2.jsonl'
output_file_path = ''
target_molecules = []
target_labels = []

data = []
for i in range(1, num_generations):
    
    new_molecules = []
    num_molecules = 20
    print("iteration", i)
    with jsonlines.open(target_file_path) as reader:
        for line in reader:
            if "\n" not in line:
                target_molecules.append(line['smiles'])
                target_labels.append(line['label'])
    unique_molecules = set() 
    
    for _ in range(num_molecules):

        target_index = random.randint(0, len(target_molecules) - 1)
        target_mol = target_molecules[target_index]
        target_label = target_labels[target_index]
        print("target, label", target_mol, target_label)
        # message = [{"role":"user", "content":f'Generating only SMILES strings, Generate a novel valid molecule similar to {target_mol} that is {target_label}-class'}]
        # response = openai.ChatCompletion.create(
        #     model = "gpt-3.5-turbo",
        #     messages = message,
        #     max_tokens=60,
        #     temperature=0.7,
        #     n=1,
        #     stop=None,
        #     timeout=60
        # )
        # new_mol = response.choices[0]
        prompt = f'Generate a novel valid molecule similar to {target_mol} that is {target_label}-class and do not generate any English text'
        response = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=60,
            temperature=0.7,
            n=1,
            stop=None,
            timeout=20
        )
        new_mol = response.choices[0].text.strip()
        print("new mol", new_mol)
        try:
            mol = Chem.MolFromSmiles(new_mol)
            if mol is not None and mol not in unique_molecules:
                #sanitized_mol = Chem.SanitizeMol(mol)
                new_molecules.append(mol)
                #print("new molecules", new_mol)
                print("new_molecules", new_molecules)
        except Exception as e:
            print(f"SMILES Parse Error: {e}. Skipping molecule: {new_mol}")
        docking_scores = []
        for mol in new_molecules:
            docking_score = calculate_docking_score(mol)
            print("docking score", docking_score)
            docking_scores.append(docking_score)

        labels = []
        mw_threshold = 700
        logp_threshold = 5
        radscore_threshold = 5
        new_target_molecules = []
        mols = []
        
        

        for mol in new_molecules:
            print("mol", Chem.MolToSmiles(mol))
            try:
                smiles = Chem.MolToSmiles(mol)
                if smiles not in unique_molecules:
                    unique_molecules.add(smiles)
                    mw = rdkit.Chem.Descriptors.MolWt(mol)
                    logp = rdkit.Chem.Descriptors.MolLogP(mol)
                    sas = sascorer.calculateScore(mol)
                    print(mw, logp, sas, )
                    if (
                        200 <= mw <= mw_threshold
                        and logp <= logp_threshold
                        and sas <= radscore_threshold
                    ):
                        labels.append('1')
                        new_target_molecules.append({'smiles': smiles, 'label': '1'})
                        #mols.append(Chem.MolToSmiles(mol))
                    else:
                        labels.append('0')
                else:
                    print("skipping duplicate molecules")
            except Exception as e:
                print(f"Molecular Property Calculation Error: {e}. Skipping molecule.")
            

        data.extend(list(zip(unique_molecules, labels)))
        
        print("data", data)


        #with jsonlines.open(target_file_path, mode='a') as writer:
            # writer.write('\\n')
            # writer.write_all(new_target_molecules)
        with jsonlines.open(target_file_path, mode='a') as writer:
            # writer.write("\\n")
            for molecule in new_target_molecules:
                writer.write(molecule)
                writer.write('\n')
        # with open(target_file_path, mode='a') as outfile:
        #     for hostDict in target_file_path:
        #         json.dump(hostDict, outfile)
        #         outfile.write('\n')

df = pd.DataFrame(data, columns=['Molecule', 'Label'])
df.to_csv(output_file_path, index=False)
# Print and analyze the results
print(f'generation {i}:')
print('generated molecules:')
print(data)
