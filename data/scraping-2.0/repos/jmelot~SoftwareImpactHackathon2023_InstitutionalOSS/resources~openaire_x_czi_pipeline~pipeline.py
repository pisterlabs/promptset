from jproperties import Properties
from utils import prepare_openaire_data
from utils import prepare_czi_data
import pandas as pd

if __name__ == "__main__":

    # # load properties file
    configs = Properties()
    with open('config.properties', 'rb') as read_prop:
        configs.load(read_prop)

    # # prepare input and output folders
    # create_folder(configs.get("INPUT_FOLDER").data)
    # create_folder(configs.get("OUTPUT_FOLDER").data)

    # # download input DOI to ROR id file from OpenAIRE
    openaire_input_file = f"{configs.get('INPUT_FOLDER').data}/{configs.get('OPENAIRE_INPUT_FILENAME').data}"

    # download_file(
    #     configs.get("OPENAIRE_DOI_TO_RORID_INPUT_FILE").data, \
    #     openaire_input_file \
    # )

    # prepare DOI to RORid relations from OpenAIRE
    openaire_parsed_file = f"{configs.get('INPUT_FOLDER').data}/{configs.get('OPENAIRE_PARSED_FILENAME').data}"
    doi_to_ror_data = prepare_openaire_data(openaire_input_file, openaire_parsed_file)
    # print(doi_to_ror_data)

    # prepare DOI to repository URL data
    czi_data_dir = f"{configs.get('INPUT_FOLDER').data}/{configs.get('CZI_DATA_DIR').data}"

    czi_raw_mentions_file = f"{czi_data_dir}/{configs.get('CZI_RAW_MENTIONS_FILE').data}"
    czi_normalized_github_repos_file = f"{czi_data_dir}/{configs.get('CZI_GITHUB_SOFTWARE_FILE').data}"

    doi_to_repo_data = prepare_czi_data(czi_raw_mentions_file, czi_normalized_github_repos_file)
    # print(doi_to_repo_data)
          
    print("Joining OpenAIRE and CZI mentions data")
         
    result_df = pd.merge(doi_to_repo_data, doi_to_ror_data, on='doi')
    result_df = result_df.drop('doi', axis=1)
    # print(result_df)

    output_file = f"{configs.get('OUTPUT_FOLDER').data}/{configs.get('OUTPUT_FILE').data}"

    print("Write retult to {output_file}")

    result_df.to_csv(output_file, index=False, sep='\t', header=True)

    print("Ready!")








