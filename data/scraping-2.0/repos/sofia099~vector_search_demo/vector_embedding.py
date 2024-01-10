import openai
import csv
import time

### EDIT BELOW
OPENAI_API_KEY = 'YOUR_OPENAI_API_KEY'
num_subsets = 9
### STOP

# set openai api key
openai.api_key = OPENAI_API_KEY

# generate embeddings
def generate_embeddings(values):
    embeddings = openai.Embedding.create(input=values, model='text-embedding-ada-002')['data'][0]['embedding']
    return embeddings

# read the csv file
def read_csv_file(file_path):
    rows = []
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows.append(row)
    return rows

# write the embeddings to the csv file
def write_embeddings_to_csv(file_path, rows):
    with open(file_path, 'w', newline='') as csvfile:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

for i in range(1,num_subsets+1):
    # read the rows from the csv file
    input_csv_file_path = f"subset_{i}.csv"
    rows = read_csv_file(input_csv_file_path)

    # generate embeddings for the values
    values = [row['movie_name']+' '+row['description'] for row in rows] # embedding the movie_name and description
    try:
        embeddings = [generate_embeddings(value) for value in values]
    except openai.error.RateLimitError:
        time.sleep(61) # sleep for at least 1min to avoid hitting RateLimitError again
        embeddings = [generate_embeddings(value) for value in values]
    except openai.error.RateLimitError:
        time.sleep(300) # sleep for 5min to avoid hitting RateLimitError again
        embeddings = [generate_embeddings(value) for value in values]

    # append the embeddings to the rows
    for row, embedding in zip(rows, embeddings):
        row['embedding'] = embedding

    write_embeddings_to_csv(input_csv_file_path, rows)

    print(f"Embeddings for subset {i} done.")