# %%

import pandas as pd
from langchain_community.document_loaders import UnstructuredMarkdownLoader
# %%
# read in csv file
csv_file = "data/Rob_Burbea_Transcripts.2023-12-31.csv"
df = pd.read_csv(csv_file)

df.columns = df.columns.str.replace(" ", "_").str.lower()


# %%
# split the transcript_or_writing column into pdf name and create new column
# remove .pdf from pdf_name
df["name"] = df.transcript_or_writing.str.split("/").str[-1].str.replace(".pdf", "")

# %%
# drop first row
df = df.drop(df.index[0])
# %%

# %%
cols = [
    "name",
    "date",
    "title_of_event",
    "title_of_talk_or_writing",
    "broad_topics",
    "detailed_topics",
    "length_of_recording",
    "type_of_recording",
]
df[cols].head()
# %%
df.type_of_recording.unique()
# %%
docs = []
total_rows = len(df[cols])
for i, row in enumerate(df[cols].iterrows(), start=1):
    metadata = dict(row[1])
    markdown_path = f"data/md_parts/{metadata['name']}.md"
    print(f"Processing {i}/{total_rows} rows")
    try:
        loader = UnstructuredMarkdownLoader(
            markdown_path, mode="elements", metadata=metadata
        )
        data = loader.load()

    except FileNotFoundError:
        print(f"File {markdown_path} not found.")
    data = loader.load()
    docs.append(data)   


# %%



# %%
# TODO split first line manually on "/\n" ?
# %%
