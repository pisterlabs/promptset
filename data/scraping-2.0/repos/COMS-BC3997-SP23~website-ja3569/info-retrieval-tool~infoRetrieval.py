import openai
import csv
from collections import Counter

openai.api_key = "" #open-ai api key

def extract_information(ngos_list):
    ngos_information = []
    engines = ["davinci-codex", "curie-codex", "babbage-codex", "ada-codex"]

    for ngo in ngos_list:
        prompt = f"Extract information for the NGO '{ngo}' accredited to participate in all future sessions of the Conference of States Parties to the CRPD. "\
                 f"Provide the following information in the format: "\
                 f"Name, official_website, country, zip code, address, sustainable_goal, list of sponsors, founding_year. "\
                 f"Ensure that the information provided is accurate and complete."

        responses = []
        for engine in engines:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                max_tokens=200,
                n=1,
                stop=None,
                temperature=0.5,
            )

            extracted_info = response.choices[0].text.strip()
            responses.append(extracted_info)

        counter = Counter(responses)
        most_common_response = counter.most_common(1)[0][0]

        ngos_information.append(most_common_response)

    return "; ".join(ngos_information)


def load_ngos_from_csv(file_path):
    ngos_list = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            ngos_list.append(row[0])
    return ngos_list


csv_file_path = "ngoList.csv"
ngos_list = load_ngos_from_csv(csv_file_path)

extracted_information = extract_information(ngos_list)
print(extracted_information)
