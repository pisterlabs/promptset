import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import numpy as np


# This script is to chunk the data we collected from PubMeb into 1500 characters chunks with 100 characters overlap and save the result in a CSV file

# Import the data that we exported from PubMed
df_data = pd.read_csv("exported_data.csv")


# Initialize langChain splitter to split abstracts into 1500 characters chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)



rows_list = []

# Iterate over the data and chunk the abstracts (add an id number for each chunk)
for index, row in tqdm(df_data.iterrows(), total=df_data.shape[0], desc='Chunking data'):    
    
    chunks = text_splitter.split_text(str(row['Abstract']))

    for i in range(len(chunks)):

        rows_list.append([row['PMID'], row['Title'], row['Abstract'], i, chunks[i], row['Key_words'], row['Authors'], row['Journal'], row['Year'], row['Month'], row['Source'], row['Country']])


df_data_chunks = pd.DataFrame(rows_list, columns=["PMID", "Title", "Abstract", "Chunk_id", "Chunk", "Key_words", "Authors", "Journal", "Year", "Month", "Source", "Country"])



# Save the chunked data in a CSV file (We save the data in chunks so we can show the progress in tqdm)
splits = np.array_split(df_data_chunks.index, 100)

for index, split in enumerate(tqdm(splits, desc='Saving to CSV')):
    if index == 0:
        df_data_chunks.loc[split].to_csv('data_chunks.csv', mode='w', index=False)
    else:
        df_data_chunks.loc[split].to_csv('data_chunks.csv', header=None, mode='a', index=False)




