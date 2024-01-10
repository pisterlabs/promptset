import openai
import jsonlines
import time
import pandas as pd
import numpy as np
import random
import json
from rdkit.Chem import Descriptors, QED
from rdkit.Chem import rdMolDescriptors as rdmd
# Set up your API key and model parameters

openai.api_key = ''
model_engine = 'text-davinci-003' # You can choose a different model if desired
temperature = 0.3 # Controls the "creativity" of the generated molecules
num_generations = 20# The number of times to generate new molecules and feed them back into the modelcond
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
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

directory_path = './LMLF/'
target_file_path = './LMLF/one-box/drd2.jsonl'
output_file_path = ''


def calculate_docking_score(smiles):
    try:
        # Generate an RDKit molecule from the SMILES string
        molecule = Chem.MolFromSmiles(smiles)
        molecule = Chem.AddHs(molecule)
    except Exception as e:
        print(f"SMILES Parse Error: {e}. Skipping molecule: {smiles}")
        return None
    if molecule is not None:
        
        try:
            AllChem.Compute2DCoords(molecule)
        except Exception as e:
            print(f"Compute2DCoords Error: {e}. Skipping molecule: {smiles}")
            return None

        
        AllChem.EmbedMolecule(molecule, AllChem.ETKDG())

        
        try:
            AllChem.MMFFOptimizeMolecule(molecule)
        except Exception as e:
            print(f"MMFFOptimizeMolecule Error: {e}. Skipping molecule: {smiles}")
            return None

        # generate a PDB file from the molecule
        pdb_filename = './ligand.pdb'
        writer = Chem.PDBWriter(pdb_filename)
        writer.write(molecule)
        writer.close()


        cmd = ['./gnina', '--config', 'DRD2_config.txt', '--ligand', './ligand.pdb', '--out', 'output.sdf', '--log', './threshold_output_log.txt', '--cpu', '4', '--num_modes', '1']
        # 
        #cmd = ['./gnina', '-r', '.4IVA.pdb', '-l', './ligand.pdb', '--autobox_ligand', './ligand.pdb', '-o', '/content/docked.txt', '--seed', '0']

        print("Docking Command:", ' '.join(cmd))
        # try:
        #     subprocess.run(cmd, check=True)
        # except subprocess.CalledProcessError as e:
        #     print("Docking Error:", e)
        #     return None
        try:
            subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print("Docking process failed:", e)
            print("Error output:", e.stderr)
            return None


        # subprocess.run(cmd, check=True)
        
        # TODO: Extract and return the docking score from the output files
        import os

        # Iterate over the files in the directory
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):  # Consider only the text files
                file_path = os.path.join(directory_path, filename)
                with open(file_path, 'r') as file:
                    lines = file.readlines()

                    for i, line in enumerate(lines):
                        if 'affinity' in line.lower() and 'cnn' in line.lower():
                            third_next_line_values = lines[i + 3].split()
                            if len(third_next_line_values) >= 4:
                                try:
                                    cnn_affinity = float(third_next_line_values[3].strip())
                                    return cnn_affinity
                                except ValueError:
                                    pass
        
        return None

def calculate_radscore(mol):
    # molecule = Chem.MolFromSmiles(mol)
    # sa_score = rdMolDescriptors.SyntheticAccessibility(molecule)
    try:
        m = Chem.MolFromSmiles(mol)
        sa_score = sascorer.calculateScore(m)
        return sa_score
    except:
        return 0
def calculate_qed_score(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is not None:
        qed_score = QED.qed(molecule)
        return qed_score
    else:
        return None
            

      



# target_mol = None
# with jsonlines.open(target_file_path) as reader:
#     for line in reader:
#         target_mol = line['smiles']
#         target_label = line['label']
        # break
target_molecules = []
target_labels = []

threshold_increment_frequency = 10
data = []
docking_threshold = 7

# Generate and feed back new molecules k times
for i in range(1, num_generations):
    if i % threshold_increment_frequency == 0 and i > 0:
        docking_threshold += 1
    
    #Generating molecules
    new_molecules = []
    num_molecules = 10
    print("iteration", i)
    with jsonlines.open(target_file_path) as reader:
        for line in reader:
            if "\n" not in line:
                target_molecules.append(line['smiles'])
                target_labels.append(line['label'])
    for _ in range(num_molecules):
        #target_mol = None
        target_index = random.randint(0, len(target_molecules) - 1)
        target_mol = target_molecules[target_index]
        target_label = target_labels[target_index]
        # with jsonlines.open(target_file_path) as reader:
        #     for line in reader:
        #         target_mol = line['smiles']
        #         target_label = line['label']
        #         break
        print("target, label", target_mol, target_label)
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
        new_molecules.append(new_mol)
        print("new molecules", new_mol)
    
    #clculate docking scores
    docking_scores = []
    for mol in new_molecules:
        docking_score = calculate_docking_score(mol)
        print("docking score", docking_score)
        docking_scores.append(docking_score)
    #calculate QED scores
    qed_scores = []
    for mol in new_molecules:
        qed_score = calculate_qed_score(mol)
        print("QED score", qed_score)
        qed_scores.append(qed_score)
    
    #calculate RADcores
    mw_scores = []
    logp_scores = []
    radscores = []
    for mol in new_molecules:
        # try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        sas = rdmd.SyntheticAccessibility(mol)
        mw_scores.append(mw)
        logp_scores.append(logp)
        radscore = calculate_radscore(mol)
        print("RD score", radscore)
        radscores.append(radscore)
        # except:
        #     continue
    
    #select molecules based on upper quartiles of docking scores and RADcores
    filtered_docking_scores = [score for score in docking_scores if score is not None]
    filtered_mw_scores = [score for score in mw_scores if score is not None]
    filtered_logp_scores = [score for score in logp_scores if score is not None]
    filtered_qed_scores = [score for score in qed_scores if score is not None]
    print("radscores", radscores)
    filtered_rad_scores = [score for score in radscores if score is not None] 
    #docking_threshold = np.percentile(filtered_docking_scores, 75)
    
    print("docking threshold", docking_threshold)
    #qed_threshold = np.percentile(filtered_qed_scores, 75)
    qed_threshold = 4
    if len(filtered_rad_scores) > 0:
        radscore_threshold = np.percentile(filtered_rad_scores, 75)
    else:
        radscore_threshold = 0.0  # Assign a default value or handle the case appropriately
    #radscore_threshold = np.percentile(filtered_rad_scores, 75)
    labels = []
    mw_threshold = 700
    logp_threshold = 6.0
    new_target_molecules = []
    for mol, docking_score, qed_score, radscore in zip(new_molecules, filtered_docking_scores, filtered_qed_scores, filtered_rad_scores):
        if (
            docking_score >= docking_threshold 
            and radscore >= radscore_threshold
            and mw <= mw_threshold
            and logp <= logp_threshold
        ):
            labels.append('1')
            new_target_molecules.append({'smiles': mol, 'label': '1'})
        else:
            labels.append('0')
    
    #Repeat or stop based on joint median change
    if i >= 1:
        prev_joint_median = np.median([max(docking_score, qed_score, radscore) for docking_score, qed_score, radscore in zip(filtered_docking_scores, filtered_qed_scores, filtered_rad_scores)])

        joint_median = np.median(filtered_docking_scores + filtered_rad_scores + filtered_qed_scores)
        if abs(joint_median - prev_joint_median) < 0.01:
            break
    

    data.extend(list(zip(new_molecules, filtered_docking_scores, filtered_qed_scores, filtered_rad_scores, labels)))
    # Perform further operations or analysis with the data as needed
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

df = pd.DataFrame(data, columns=['Molecule', 'Docking Score', 'QED Score', 'RADscore', 'Label'])
df.to_csv(output_file_path, index=False)
joint_scores = filtered_docking_scores + filtered_rad_scores + filtered_qed_scores
# Print and analyze the results
print(f'generation {i}:')
print(f'joint median: {joint_median}')
print('generated molecules:')
print(data)
