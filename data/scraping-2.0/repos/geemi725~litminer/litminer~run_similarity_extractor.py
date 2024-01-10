import os
import openai

keys_file = open("/home/wellawatte/Desktop/key.txt")
lines = keys_file.readlines()
apikey = lines[0].rstrip()
os.environ["OPENAI_API_KEY"] = apikey
openai.api_key = apikey
import Similarity_score_GPT as sim

llm_model = "gpt-4"
queries = [
    """What is the type of membrane/membrane material used in this work? 
Pick one answer from the following list: \n
Facilitated transport, \n
polymer of intrinsic porosity, \n
thermal rearrangement polymers, \n
Metal-induced ordered microporous polymers, \n
Carbon molecular sieve, \n
zeolite \n,
MOF-membranes""",
    """What is the maximum CO2 permeability(permeance) of the membrane for CO2/N2 mixture in this work? DO NOT provide values for other mixtures such as CO2/CH4.
Generally, CO2 permeance is reported in GPU or mol m^-2 s^-1 Pa^-1 units. eg: 2.06 x 10-7 mol m-2 s-1 Pa-1""",
    """What is the CO2/N2 selectivity for the given permeability reported in this study? \n
You may find the permeability value in the previous Q&A section.
Do not provide values from other studies. Provide only the value. Selectivity is also referred to as separation factor. """,
]

file_dir = "/home/wellawatte/Documents/MOF_grant/lit_extraction/literature/membranes"
save_dir = "/home/wellawatte/Documents/litminer/litminer/data/logs"


print("running only similarity search only extractor")
for i in range(1, 6):
    try:
        df = sim.run_extractor(
            file_dir=file_dir,
            queries=queries,
            llm_model=llm_model,
            logfile=f"{save_dir}/log_SIMscore{i}.txt",
            headers=[
                "membrane",
                "excerpt",
                "score",
                "permeability",
                "excerpt",
                "score",
                "selectivity",
                "excerpt",
                "score",
            ],
        )
        df.to_csv(
            f"data/SIM_score_runs/{llm_model.upper()}_SIM_scores_{i}.csv",
            index=False,
            escapechar="\\",
            sep=",",
        )
    except:
        print(f"{llm_model}, {i}, error!!")
        continue
