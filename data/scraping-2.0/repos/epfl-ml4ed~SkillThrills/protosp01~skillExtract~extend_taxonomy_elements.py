# %%
import os, re
import argparse
import pandas as pd
import openai
import os
import pandas as pd
import re
from split_words import Splitter

# %%
from utils import *
from prompt_template import PROMPT_TEMPLATES
from api_key import *

# %%


def extend_taxomony():
    # Navigating to the folder where the data is stored
    os.chdir("../data/taxonomy/")
    assert os.getcwd().split("/")[-1] == "taxonomy", "check path"
    # %%

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate_new", action="store_true", help="Whether to generate new alternative names or not")
    parser.add_argument("--no_inter", action="store_true", help="Whether to not save intermediate raw files")

    args = parser.parse_args()
    # fmt: on

    generator_models = {
        "chatgpt": "gpt-3.5-turbo",
        "gpt-4": "gpt-4",
    }

    # splitter = Splitter()

    openai.api_key = API_KEY

    engine = "chatgpt"

    PROMPT = f"I am looking for occurrences of the <SKILL_TYPE> '<NAME>' in a document. However, the author doesn't always refer to this <SKILL_TYPE> using the full name. Generate only a list of exactly 10 other names that I could look for, separated by commas."

    EXAMPLE_DICT = {
        "technologies": [
            "Microsoft Excel",
            "Excel, MS Excel, Microsoft Excel, Spreadsheet software by Microsoft, Microsoft's spreadsheet application, Excel program, Excel software, Microsoft's data analysis tool, Microsoft's workbook software, Spreadsheet program by Microsoft",
        ],
        "certifications": [
            "AWS DevOps Engineer",
            "AWS, AWS DevOps Specialist, Amazon DevOps Engineer, AWS DevOps Practitioner, Certified AWS DevOps Professional, AWS DevOps Architect, Amazon Web Services DevOps Expert, AWS DevOps Solutions Engineer, AWS Cloud DevOps Engineer, AWS DevOps Deployment Specialist, AWS DevOps Integration Engineer",
        ],
    }
    # TODO: maybe reformat above to be included in prompt_template.py and refactor below to be included in utils.py

    class Generator:
        def __init__(self):
            self.model = generator_models[engine]

        def generate(self, skill_type, skill_name):
            messages = []
            sys_message = {
                "role": "system",
                "content": f"You are an expert at human resources, specialized in the IT domain.",
            }
            messages.append(sys_message)

            # Get the prompt

            question_example_content = PROMPT.replace(
                "<SKILL_TYPE>", skill_type
            ).replace("<NAME>", EXAMPLE_DICT[skill_type][0])
            example_answer = EXAMPLE_DICT[skill_type][1]
            question_content = PROMPT.replace("<SKILL_TYPE>", skill_type).replace(
                "<NAME>", skill_name
            )
            messages.append({"role": "user", "content": question_example_content})
            messages.append({"role": "assistant", "content": example_answer})
            messages.append({"role": "user", "content": question_content})
            flag = True
            while flag:  # while getting exception from API, retry
                try:
                    response = openai.ChatCompletion.create(
                        model=self.model,
                        messages=messages,
                        top_p=1.0,
                        temperature=0.8,
                        frequency_penalty=0.8,
                        presence_penalty=0.5,
                    )
                    flag = False
                except Exception as e:
                    print(
                        f"The following error has happened. Waiting for 5seconds:\n{e}"
                    )
                    time.sleep(5)
            output_text = response["choices"][0]["message"]["content"]
            return output_text

        # added function to be able to apply to the dataframe instead of the column
        def get_alt_names(self, row):
            alt_name = self.generate(row["Level 1"].lower(), row["Level 2"])
            return alt_name

    generator = Generator()

    # %%
    tech_certif_lang = pd.read_csv("tech_certif_lang.csv")
    certif = (
        tech_certif_lang[tech_certif_lang["Level 1"] == "Certifications"]
        .copy()
        .reset_index(drop=True)
    )  # making a copy to avoid working on splices and reseting index to avoid problems

    tech = (
        tech_certif_lang[tech_certif_lang["Level 1"] == "Technologies"]
        .copy()
        .reset_index(drop=True)
    )

    # %%

    def generate_alt_names(df, skill_type):
        print(f"generating alternative names for {skill_type}")
        df["alternative_names"] = df.apply(generator.get_alt_names, axis=1)
        if not args.no_inter:
            df.to_csv(
                f"{skill_type}_alternative_names_raw.csv",
                index=False,
            )
        return df

    def load_alt_names(df, skill_type):
        try:
            df = pd.read_csv(f"{skill_type}_alternative_names_raw.csv")
            print(f"loaded raw {skill_type} alternative names file")
        except:
            print(f"raw file not found, generating alternative names for {skill_type}")
            df["alternative_names"] = df.apply(generator.get_alt_names, axis=1)
            if not args.no_inter:
                df.to_csv(
                    f"{skill_type}_alternative_names_raw.csv",
                    index=False,
                )
        return df

    if args.generate_new:
        certif = generate_alt_names(certif, "certifications")
        tech = generate_alt_names(tech, "technologies")
    else:
        certif = load_alt_names(certif, "certifications")
        tech = load_alt_names(tech, "technologies")

    # %%
    # adding smaller names to alternative names clean
    pattern = r"\((.*?)\)"

    def get_name(certif):
        matches = re.findall(pattern, certif)
        if matches:
            return matches[0]
        else:
            return ""

    smaller_name = [get_name(name) for name in list(certif["Level 1.5"])]
    smaller_name2 = [get_name(name) for name in list(certif["Level 2"])]

    certif["alternative_names"] = (
        certif["alternative_names"] + ", " + smaller_name + ", " + smaller_name2
    )

    # %%
    print("cleaning alternative names for certifications")
    certif["alternative_names_clean"] = certif.apply(
        lambda row: clean_skills_list(row["Level 2"], row["alternative_names"]), axis=1
    )
    # %%

    certif = certif[["unique_id", "Level 2", "alternative_names_clean"]]
    certif.to_csv("certifications_alternative_names.csv", index=False, sep="\t")
    print("saved certifications")

    # %%
    print("cleaning alternative names for technologies")
    tech["alternative_names_clean"] = tech.apply(
        lambda row: clean_skills_list(row["Level 2"], row["alternative_names"]), axis=1
    )

    tech = tech[["unique_id", "Level 2", "alternative_names_clean"]]
    tech.to_csv("technologies_alternative_names.csv", index=False, sep="\t")
    print("saved technologies")


if __name__ == "__main__":
    extend_taxomony()
