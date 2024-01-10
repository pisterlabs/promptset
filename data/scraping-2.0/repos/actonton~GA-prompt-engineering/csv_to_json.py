import csv
import json

file = 'imdb_movies.csv'
json_file = 'imdb_movies.json'


#Read CSV File
def read_CSV(file, json_file):
    csv_rows = []
    with open(file) as csvfile:
        reader = csv.DictReader(csvfile)
        field = reader.fieldnames
        for row in reader:
            csv_rows.extend([{field[i]:row[field[i]] for i in range(len(field))}])
        convert_write_json(csv_rows, json_file)

#Convert csv data into json
def convert_write_json(data, json_file):
    with open(json_file, "w") as f:
        f.write(json.dumps(data, sort_keys=False, indent=4, separators=(',', ': '))) #for pretty
        f.write(json.dumps(data))


#read_CSV(file,json_file)

with open('./data/imdb_movies.json') as user_file:
  file_contents = user_file.read()


with open('./data/filtered_movies.json') as user_file:
  filtered_movies = user_file.read()

from langchain.document_loaders import JSONLoader



#loader = JSONLoader("./filtered_movies.json")
loader = JSONLoader('./data/filtered_movies.json', jq_schema=".[]", text_content=False)
documents = loader.load()
# print(documents)
#text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

#docs = text_splitter.split_documents(documents)
