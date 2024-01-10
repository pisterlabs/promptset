import argparse
import csv
import json
import os
import pandas as pd


def reformat_orca_url_matches(url_matches: str, full_data: str) -> list:
    """
    Reformat url matches over the ORCA data into the standard format
    :param url_matches: Name of file containing owner <-> ROR matches found from URL match over ORCA data
    :param full_data: Full ORCA data download
    :return: List of reformatted records
    """
    org_to_repos = {}
    reformatted = []
    with open(full_data) as f:
        for line in f:
            js = json.loads(line)
            owner = js["owner_name"]
            repo = js["current_name"]
            if owner not in org_to_repos:
                org_to_repos[owner] = []
            org_to_repos[owner].append(repo)
    with open(url_matches) as f:
        org_to_ror = json.loads(f.read())
    for owner in org_to_repos:
        unique_repos = set(org_to_repos[owner])
        for repo in unique_repos:
            name = f"{owner}/{repo}"
            for ror_id in org_to_ror.get(owner, []):
                reformatted.append({
                    "software_name": name,
                    "github_slug": name,
                    "ror_id": ror_id,
                    "extraction_method": "url_matches"
                })
    return reformatted


def reformat_stack_readme_matches(stack_matches: str) -> list:
    """
    Reformat NER matches over the Stack README data into the standard format
    :param url_matches: Name of file containing software-ROR affiliations extracted from The Stack
    :return: List of reformatted records
    """
    reformatted = []
    with open(stack_matches) as f:
        reader = csv.DictReader(f)
        for line in reader:
            ror_id = line["ror_id"]
            if ror_id:
                reformatted.append({
                    "software_name": line["repo_name"],
                    "github_slug": line["repo_name"],
                    "ror_id": ror_id,
                    "extraction_method": "ner_text_extraction"
                })
    return reformatted


def reformat_working_curated(working_curated: str) -> list:
    """
    Reformat working curated data into the standard format
    :param working_curated: Name of file containing minimal working curated data
    :return: List of reformatted records
    """
    reformatted = []
    with open(working_curated) as f:
        reader = csv.DictReader(f)
        for row in reader:
            reformatted.append({
                "software_name": row["software_name"],
                "github_slug": row["github_slug"],
                "ror_id": row["ror_id"],
                "extraction_method": row["extraction_methods"]
            })
    return reformatted


def reformat_czi_affiliation_rors(software_to_rors: str) -> list:
    """
    Reformat csvs mapping affiliation software to ror ids into the standard format
    :param software_to_rors: Name of file containing software to author affiliation rors
    :return: List of reformatted records
    """
    reformatted = []
    with open(software_to_rors) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ror = row["ROR_ID"]
            repo = row["github_slug"]
            repo = "/".join(repo.split("/")[-2:])
            if not ror or ror == "NA":
                continue
            reformatted.append({
                "software_name": row["mention"],
                "github_slug": repo,
                "ror_id": ror,
                "extraction_method": "czi_affiliation_links"
            })
        return reformatted


def reformat_joss_affiliation_rors(software_to_rors: str) -> list:
    """
    Reformat jsonl mapping affiliation software to ror ids into the standard format
    :param software_to_rors: Name of file containing software to author affiliation rors
    :return: List of reformatted records
    """
    reformatted = []
    with open(software_to_rors) as f:
        for line in f:
            row = json.loads(line)
            ror = row["ror_id"]
            repo = row["github_slug"]
            reformatted.append({
                "software_name": row["software_name"],
                "github_slug": repo,
                "ror_id": ror,
                "extraction_method": "joss_affiliation_links"
            })
        return reformatted


def reformat_openaire_czi_matches(openaire_czi_matches_file: str):
    """
    Reformat csv.gz from OpenAIRE and CZI mentions merged data
    :param openaire_czi_matches_file: Name of file containing software to ror id relations
    :return: List of reformatted records
    """
    df = pd.read_csv(openaire_czi_matches_file, compression='gzip', delimiter='\t', encoding='utf-8')
    df = df.rename(columns={'github_repo': 'github_slug', 'software': 'software_name', 'RORid': 'ror_id'})
    df = df.drop_duplicates()
    df['extraction_method'] = 'openaire_czi'

    return df.to_dict(orient='records')


def merge_rows(datasets: list) -> list:
    """
    Merge data across disparate sources, with one row per software-ROR pair
    :param datasets: List of lists of records in the format shown in `reformat_orca_url_matches`, one per data source
    :return: List of deduplicated records
    """
    id_to_record = {}
    # we are iterating through a list of records containing our output rows (see format in `reformat_orca_url_matches`).
    # We're creating an id for each row based on the name of the software and the ror id. If this id is present in
    # `id_to_record`, we will add the extraction method to the existing list. If not, we will add a new
    # record to `id_to_record`
    for dataset in datasets:
        for row in dataset:
            row["software_name"] = row["software_name"].strip()
            id = f"{row['software_name']}/{row['ror_id']}".lower()
            extraction_method = row["extraction_method"]
            if id in id_to_record:
                if extraction_method not in id_to_record[id]["extraction_methods"]:
                    id_to_record[id]["extraction_methods"].append(extraction_method)
            else:
                row["extraction_methods"] = [row.pop("extraction_method")]
                id_to_record[id] = row
    merged = []
    # Not casting aspersions, just noting that some methods return less ambiguous matches than others!
    high_quality_methods = ["czi_affiliation_links", "joss_affiliation_links", "by_name", "human_curated"]
    medium_quality_methods = ["ner_text_extraction", "url_matches"]
    for _, record in id_to_record.items():
        methods = record["extraction_methods"]
        is_high_quality = len(methods) > 1 or any([m in high_quality_methods for m in methods])
        is_medium_quality = len(methods) == 1 and any([m in medium_quality_methods for m in methods])
        record["quality"] = 1 if is_high_quality else (0.5 if is_medium_quality else 0)
        merged.append(record)
    merged.sort(key=lambda row: f"{row['software_name']}/{row['ror_id']}".lower())
    return merged


def write_reformatted(orca_url_matches: str, orca_data: str, stack_readme_matches: str, working_curated: str,
                      czi_software_rors: str, joss_software_rors: str, openaire_czi_matches: str, output_csv: str,
                      output_json: str):
    """
    Merge data from disparate sources and write out in a single CSV
    :param orca_url_matches: matches from repo owner urls to ROR urls
    :param orca_data: ORCA data download, containing additional metadata for each ORCA repo
    :param stack_readme_matches: Affiliations extracted from The Stack readmes using NER
    :param working_curated: Curated data, augmented with matches based on ROR API
    :param czi_software_rors: RORs pulled from author affiliations from CZI software-repo links
    :param joss_software_rors: RORS pulled from author affiliations in JOSS software-repo links
    :param openaire_czi_affiliations: ROR id to repository URLs by joining OpenAIRE ROR-to-DOI affiliations
           and CZI DOI-to-software mentions
    :param output_csv: File where output csv should be written
    :param output_json: File where output json should be written
    :return: None
    """
    orca = reformat_orca_url_matches(orca_url_matches, orca_data)
    stack_readme = reformat_stack_readme_matches(stack_readme_matches)
    working = reformat_working_curated(working_curated)
    czi_affiliations = reformat_czi_affiliation_rors(czi_software_rors)
    joss_affiliations = reformat_joss_affiliation_rors(joss_software_rors)
    openaire_czi_matches = reformat_openaire_czi_matches(openaire_czi_matches)
    # To add another dataset, write a reformat_<your data> function to put data in the format shown in
    # `reformat_orca_url_matches`, then put the output in the array below
    merged_rows = merge_rows([orca, stack_readme, working, czi_affiliations, joss_affiliations, openaire_czi_matches])

    rors = set()
    software = set()
    with open(output_csv, mode="w") as f:
        writer = csv.DictWriter(f, fieldnames=["software_name", "github_slug", "ror_id", "extraction_methods",
                                               "quality"])
        writer.writeheader()
        for row in merged_rows:
            rors.add(row["ror_id"])
            if row["github_slug"]:
                software.add(row["github_slug"])
            row["extraction_methods"] = ";".join(row["extraction_methods"])
            writer.writerow(row)
    print(f"Wrote {len(merged_rows)} software-ror links containing {len(rors)} distinct ROR ids and "
          f"{len(software)} distinct GitHub repositories")

    ror_to_software = {}
    for row in merged_rows:
        ror = row["ror_id"]
        if ror not in ror_to_software:
            ror_to_software[ror] = {}
        ror_to_software[ror][row["software_name"]] = {
            "github_slug": row["github_slug"],
            "extraction_methods": row["extraction_methods"],
            "quality": row["quality"]
        }
    with open(output_json, mode="w") as f:
        f.write(json.dumps(ror_to_software, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orca_url_matches",
                        default=os.path.join("github_org_url_matching_pipeline", "orca_org_rors.json"))
    parser.add_argument("--orca_data", default=os.path.join("github_org_url_matching_pipeline", "orca_download.jsonl"))
    parser.add_argument("--stack_readme_affiliations",
                        default=os.path.join("ner_text_extraction_pipeline", "links.csv"))
    parser.add_argument("--working_curated", default=os.path.join("scicrunch", "scicrunch_working_file_minimal.csv"))
    parser.add_argument("--czi_software_rors",
                        default=os.path.join("czi_affiliation_links_pipeline", "links_with_slugs.csv"))
    parser.add_argument("--joss_software_rors", default=os.path.join("joss_affiliations", "joss_rors.jsonl"))
    parser.add_argument("--openaire_czi_matches",
                        default=os.path.join("openaire_x_czi_pipeline", "output", "openaire_x_czi_out.csv.gz"))

    # add more arguments to ingest more data sources
    parser.add_argument("--output_csv", default=os.path.join("..", "software_to_ror.csv"))
    parser.add_argument("--output_json", default=os.path.join("..", "software_to_ror.json"))
    args = parser.parse_args()

    write_reformatted(args.orca_url_matches, args.orca_data, args.stack_readme_affiliations, args.working_curated,
                      args.czi_software_rors, args.joss_software_rors, args.openaire_czi_matches, args.output_csv,
                      args.output_json)
