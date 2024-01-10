from glob import glob
from pprint import pprint

import pandas

from llm_projects.healthcare import (
    get_admissions_statistics,
    get_diagnosis_statistics,
    get_labitem_statistics,
    get_omr_statistics,
    get_patient_statistics,
    get_services_statistics,
)
from llm_projects.healthcare.text_data_analyses.embedding import embedder_test

from .complex_analysis import admission_to_transfer


def read_csvs(folder):
    files = glob(folder + "*.csv")
    df_dict = {}
    for idx, file in enumerate(files):
        filename = file.split("/")[-1].split(".")[0]

        from langchain.document_loaders.csv_loader import CSVLoader

        df = pandas.read_csv(file, nrows=2)
        print(df.columns)
        loader = CSVLoader(
            file_path=file,
            # csv_args={
            #     "delimiter": ",",
            #     "quotechar": '"',
            #     # "fieldnames": columns,
            # },
            metadata_columns=[
                "note_id", "subject_id", "hadm_id", "charttime", "storetime"
            ],
        )
        data = loader.load()

        embedder_test(data)
        # print(idx, filename, len(df.columns), df.shape)
        # print(df.columns)
        # pprint(df.head(5).to_dict())

        print("\n")
        # print(df.columns)
        # print()
        df_dict[filename] = df
        # get_embeddings(df)
    pass


def read_csvs_old(folder):
    files = glob(folder + "*.csv")

    completed = [
        "patients",
        "admissions",
        "provider",
        "services",
        "d_labitems",
        "omr",
    ]

    skipped_for_now = ["d_hcpcs"]

    df_dict = {}
    for idx, file in enumerate(files):
        filename = file.split("/")[-1].split(".")[0]
        if filename not in completed + skipped_for_now:
            continue
        # if filename in skipped_for_now:
        #     continue
        #
        # if filename != "emar_detail":
        #     continue
        # df = pandas.read_csv(file, nrows=1000)
        # get_omr_statistics(df)
        df = pandas.read_csv(file, nrows=100)
        print(idx, filename, len(df.columns), df.shape)
        print(df.columns)
        pprint(df.head(5).to_dict())

        print("\n\n")
        # print(df.columns)
        # print()
        df_dict[filename] = df

    pass


# small files, single column
# provider_df = pandas.read_csv(files[1])  # no information, only provider_id

# large files
# pharmacy_df = pandas.read_csv(files[2], nrows=10000)


# completed functions

# admissions_df = pandas.read_csv(files[0])
# patient_df = pandas.read_csv(files[18])

# get_patient_statistics(df_dict, patient_df)
# get_admissions_statistics(admissions_df)


# services_df = pandas.read_csv(files[15])  # no information, only provider_id
# get_services_statistics(services_df)
