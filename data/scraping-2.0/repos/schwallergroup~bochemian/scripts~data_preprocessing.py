import yaml
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import pubchempy as pcp
import pandas as pd
import openai
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from functools import partial
from bochemian.data.procedures import get_component_details, generate_procedure
import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from bochemian.gprotorch.data_featuriser.featurisation import (
    ada_embeddings,
    gte_embeddings,
    bge_embeddings,
    instructor_embeddings,
    e5_embeddings,
)


# Initialize the argument parser
parser = argparse.ArgumentParser(description="Generate procedures and featurizations.")

# Add arguments
parser.add_argument(
    "--data_path",
    type=str,
    help="Path to the CSV file with the reactions data",
    default="../data/reactions/bh/bh_reaction_1.csv",
)
parser.add_argument(
    "--config_path",
    type=str,
    help="Path to the YAML file with the procedure template and details",
    default="../templates/basic.yaml",
)
# parser.add_argument(
#     "output_path",
#     type=str,
#     help="Path to save the new CSV file with the procedure texts",
# )
parser.add_argument(
    "--featurization",
    type=str,
    default=None,
    help="The type of featurization to apply (if any)",
)

# Parse the arguments
args = parser.parse_args()

# Load the YAML configuration file
print(args.config_path)
with open(args.config_path) as file:
    config = yaml.full_load(file)

# Load the CSV file with the reactions data
print(args.data_path)
data = pd.read_csv(args.data_path)

# Get unique SMILES strings
unique_smiles = set()
for component in config["components"]:
    unique_smiles.update(data[component].tolist())


# Get details for all unique SMILES strings
smiles_details = {smiles: get_component_details(smiles) for smiles in unique_smiles}

# Define the procedure template
procedure_template = config["template"]


# Partially apply the generate_procedure function with the extra arguments
details = config.get("details")
if details is None:
    details = []

partial_generate_procedure = partial(
    generate_procedure,
    components=config["components"],
    details=details,
    procedure_template=procedure_template,
    smiles_details=smiles_details,
)

# Apply the function to generate procedures
with ProcessPoolExecutor() as executor:
    data["procedure"] = list(
        tqdm(
            executor.map(partial_generate_procedure, data.to_dict("records")),
            total=len(data),
            desc="Generating Procedures",
        )
    )


embedding_functions = [
    ada_embeddings,
    bge_embeddings,
    gte_embeddings,
    instructor_embeddings,
    e5_embeddings,
]

# Apply each embedding function dynamically
for embed_func in embedding_functions:
    column_name = embed_func.__name__
    embeddings = embed_func(data["procedure"].tolist())
    data[column_name] = embeddings.tolist()

# Extract the config name from the config path
config_name = os.path.basename(args.config_path).replace(".yaml", "")

# Extract the dataset name from the data path
data_name = os.path.basename(args.data_path).split(".")[0]

# Create the output file name using the config and dataset names, and the timestamp
output_file_name = f"{data_name}_procedure_template_{config_name}.csv"

# Create the full output path using the output directory and the output file name
output_path = os.path.join("../data/processed", output_file_name)

# Save the new CSV file with the procedure texts
data.to_csv(output_path, index=False)
