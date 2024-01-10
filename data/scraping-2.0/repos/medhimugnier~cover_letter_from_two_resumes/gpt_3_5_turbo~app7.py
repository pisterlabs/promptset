import csv
import openai
from datetime import datetime
import yaml

# Load configuration from the config file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Set the OpenAI API key
openai.api_key = config["openai"]["api_key"]

# Read candidate and manager data from CSV files
candidate_file = "data/input/candidates_annotated.csv"
manager_file = "data/input/managers_annotated.csv"

# Generate cover letters for each candidate and manager
with open(candidate_file, "r", encoding="utf-8") as candidate_csv, \
        open(manager_file, "r", encoding="utf-8") as manager_csv:

    candidate_reader = csv.DictReader(candidate_csv)
    manager_reader = csv.DictReader(manager_csv)

    for candidate_row in candidate_reader:
        # Extract relevant information from the candidate row
        candidate_last_name = candidate_row["lastname"]

        for manager_row in manager_reader:
            # Extract relevant information from the manager row
            manager_last_name = manager_row["lastname"]
            company_name = manager_row["COMPANYNAME"]

            # Construct the prompt for OpenAI API
            prompt = f"""Vous êtes un candidat à la recherche d'une alternance, et vous rédigez une lettre de motivation spontanée (email) 
            pour exprimer votre intérêt à travailler avec {manager_row['FIRSTNAME']} {manager_last_name}, et son équipe chez {company_name}. 
            Mettez en avant l'adéquation de vos compétences, de votre expérience et comment vous pouvez contribuer à leur société.
            Ne mettez pas d'en-tête, d'objet, de destinataire. Uniquement le corps de l'email/la candidature spontanée.
            Signez par votre nom de famille, éventuellement de votre prénom, adresse email et numéro de téléphone.
            Ne mettez pas de variables, uniquement des valeurs. 
            Le message rédigé n'est pas un template, il sera directement envoyé à un manager.\n\n"""
            prompt += "Profil du candidat :\n"
            prompt += "\n".join(f"{column}: {value}" for column, value in candidate_row.items())
            prompt += "\n\n"
            prompt += "Profil de l'entreprise :\n"
            prompt += f"Nom de l'entreprise: {company_name}\n"
            prompt += "\n".join(f"{column}: {value}" for column, value in manager_row.items())

            # Print the prompt before sending to OpenAI
            print("Prompt:")
            print(prompt)
            print()

            # Generate a response using GPT-3.5-turbo
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                ],
                max_tokens=500,
                temperature=0.7,
            )

            # Extract the generated cover letter from the response
            cover_letter = response.choices[0].message.content.strip()

            # Generate output filename based on last names and current timestamp
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            output_file = f"data/output/{timestamp}_{candidate_last_name}_{manager_last_name}.txt"

            # Write the generated cover letter to the output file
            with open(output_file, "w", encoding="utf-8") as file:
                file.write(cover_letter)

            print("Cover letter generated and saved:", output_file)
            print()

        # Reset the file pointer to the beginning for the next candidate
        manager_csv.seek(0)
