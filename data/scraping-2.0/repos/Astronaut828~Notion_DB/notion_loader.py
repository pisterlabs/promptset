import csv
import glob
from langchain.document_loaders import NotionDirectoryLoader

# Specify the path to the Notion database dump directory
notion_db_directory = "Notion_DB/Notion_Dump"

# Create a NotionDirectoryLoader instance
loader = NotionDirectoryLoader(notion_db_directory)

# Load data from the Notion database dump
docs = loader.load()

# Process and work with the loaded data as needed
print(docs) # This will be used as input for the next step

# Find all CSV files in the directory
csv_files = glob.glob(f"{notion_db_directory}/*.csv")

# Open and read each CSV file
for csv_file in csv_files:
    with open(csv_file, mode='r', encoding='utf-8-sig') as file:
        csv_reader = csv.reader(file, delimiter=';')
        # Print the header (column names)
        headers = next(csv_reader)
        print(f"Columns: {headers}")
        # Print each row of data
        for row in csv_reader:
            data = dict(zip(headers, row))
            print(data) # This will be used as input for the next step