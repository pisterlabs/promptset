import openai
import requests
import os
from Bio.Seq import Seq
from Bio import AlignIO
from io import StringIO

def ai_generate_sequence_list(prompt:str):

    with open("openai-key.txt", 'r') as file:
        OPENAI_API_KEY = file.read().replace("\n", '')
        
    openai.api_key = OPENAI_API_KEY

    messages = [
        {"role": "system", "content": "For the following protein, create a .yml file containing a single protein per protein domain with the following format. - domain: type: name: pfam:"},
        {"role": "user", "content": prompt}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4",
        max_tokens=512,
        temperature=0,
        messages=messages
    )

    pfam_values = []
    choices = response["choices"]
    for choice in choices:
        message = choice["message"]
        content = message["content"]
        yaml_data = content.split("\n")[3:]
        for line in yaml_data:
            if "pfam" in line:
                try:
                    pfam_value = line.split(": ")[1]
                    pfam_values.append(pfam_value)
                except IndexError:
                    print("Warning: Unexpected data format. Skipping.")
                    
    # Create a set of unique pfam_values to prevent duplicate requests
    pfam_values = list(set(pfam_values))

    protein_sequences_seq = seq_seed_gen(pfam_values=pfam_values)

    return protein_sequences_seq
    
def fp_generate(protein_sequences_seq: list):
    if len(protein_sequences_seq) == 0:
        print("No protein sequences!")
        return Seq("")

    fusion_prot = ""
    protein_sequences = []
    
    for i in range(len(protein_sequences_seq)):
        protein_sequences_seq[i] = list(str(protein_sequences_seq[i]))

        for j in range(len(protein_sequences_seq[i])):
            for element in protein_sequences_seq[i]:
                if element == "-" or element == ".":
                    protein_sequences_seq[i].remove(element)


        protein_sequences.append(Seq("".join(protein_sequences_seq[i])))

    for protein_sequence in protein_sequences:
        fusion_prot+=protein_sequence + "GSGSGS"

    fusion_prot = fusion_prot[0:fusion_prot.rfind("GSGSGS")]

    if fusion_prot[0] != "M":
        fusion_prot = "M" + fusion_prot
    
    fusion_prot_seq = Seq(fusion_prot)
    return(fusion_prot_seq)

def seq_seed_gen(pfam_values: list):
    protein_sequences = []
    #protein_seed_sequences = []
    url = "https://www.ebi.ac.uk/interpro/api/entry/pfam/"
    for pfam_value in pfam_values:
        response = requests.get(url + pfam_value + "?annotation=alignment:seed")
        if response.status_code == 200:
            align = AlignIO.read(StringIO(response.text), "stockholm")
            protein_seed_sequences = [record.seq for record in align]

            protein_sequence = create_sequence_based_on_common(protein_seed_sequences)
            protein_sequences.append(protein_sequence)
            #protein_seed_sequences = []
        else:
            print(f"Warning: Failed to retrieve data for pfam_value {pfam_value}.")
    
    return protein_sequences

def seq_logo_gen(pfam_values: list):
    protein_sequences  = []
    url = "https://www.ebi.ac.uk/interpro/api/entry/pfam/"
    data_list = []
    for pfam_value in pfam_values:
        response = requests.get(url + pfam_value + "?annotation=logo")
        if response.status_code == 200:
            sequence_response = response.json()
            data_list.append(sequence_response["probs_arr"])
        else:
            print(f"Warning: Failed to retrieve data for pfam_value {pfam_value}.")

    alphabets_list = []
    for data in data_list:
        last_strings = [lst[-1] for lst in data]
        alphabets_list.append(s.split(':')[0] for s in last_strings)

    protein_sequences = []
    for sequence_arr in alphabets_list:
        protein_sequences.append(Seq("".join(sequence_arr)))
    
    return protein_sequences

def create_sequence_based_on_common(sequences):
    # Assumption! All sequences are the same length
    length = len(sequences[0])

    # Find the most common letter at each position
    # If there is a tie, choose the first one    
    consensus = ""
    for i in range(length):
        # Get the ith letter from each sequence
        letters = [seq[i] for seq in sequences]
        # Find the most common letter
        most_common = max(set(letters), key=letters.count)
        consensus += most_common

    return consensus

def rank_sequences_based_on_common(sequences):
    # Assumption! All sequences are the same length
    length = len(sequences[0])

    # Find the most common letter at each position
    # If there is a tie, choose the first one    
    consensus = ""
    for i in range(length):
        # Get the ith letter from each sequence
        letters = [seq[i] for seq in sequences]
        # Find the most common letter
        most_common = max(set(letters), key=letters.count)
        consensus += most_common

    # Rank the sequences based on how many letters they have in common with the consensus
    ranked_sequences = []
    for seq in sequences:
        score = 0
        for i in range(length):
            if seq[i] == consensus[i]:
                score += 1
        ranked_sequences.append((score, seq))

    # Sort the sequences by their score
    ranked_sequences.sort(reverse=True)

    return ranked_sequences
