import openai, pinecone, os, re
from tqdm.auto import tqdm
from unidecode import unidecode


def read_text_files(directory):
    """Read all text files in a given directory and return their content."""
    texts = []  # list of strings
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r') as file:
                texts.append(file.read())
    return texts


def clean_text(
        text):  # cleans the text, good practice, doesn't hurt to have. gurantees it will be just words and numbers
    # Remove timestamps
    cleaned_text = re.sub(r'\d{1,2}:\d{2}', '', text)

    # Remove sequences of numbers + non-alphanumeric characters
    cleaned_text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Remove sequences of single characters separated by spaces
    cleaned_text = re.sub(r'(?<=\s)[a-zA-Z](?=\s)', '', cleaned_text)

    # Remove sequences that don't seem to form meaningful words (this is a heuristic and might need adjustments)
    cleaned_text = re.sub(r'\b[a-zA-Z]{1,2}\b', '', cleaned_text)

    # Remove strings that are too short
    cleaned_text = ' '.join([word for word in cleaned_text.split()]) #if len(word) > 1])

    ascii_text = unidecode(cleaned_text)

    # Remove non-alphanumeric characters except for spaces
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', ascii_text)
    print(cleaned_text)
    return cleaned_text


def convert_to_ascii_and_remove_special_chars(text):
    # Convert to closest ASCII representation
    ascii_text = unidecode(text)

    # Remove non-alphanumeric characters except for spaces
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', ascii_text)

    return cleaned_text


def split_text_into_chunks(text, chunk_size):  # you have a very long file, and you want to split the doc up into chunks
    clean_words = clean_text(text)
    words = clean_words.split()  # Split the text by whitespace, now a python list where each word is its own element
    chunks = []

    for i in range(0, len(words), chunk_size):  # iterate through list to break it into chunk size loosely 15 words
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)


    print (chunks[0])
    return chunks


def add_text_to_db():
    # Specify the directory containing text files
    directory = "/Users/juju/Desktop/What would Alix Earle Do/transcripts/"

    # Read text files content
    context_texts = read_text_files(directory)
    counter = 0
    for doc in context_texts:
        counter += 1
        print('working on doc: {}/{}'.format(counter, len(context_texts)))

        MODEL = "text-embedding-ada-002"
        res = openai.embeddings.create(
            model=MODEL,
            input=[
            "Sample document text goes here",
            "there will be several phrases in each batch"
            ],
            encoding_format="float"
        )
        embeds = [record.embedding for record in res.data]

        # initialize connection to pinecone
        pinecone.init(
            api_key=os.getenv('PINECONE_API_KEY'),
            environment="gcp-starter"
        )

        # check if 'openai' index already exists (only create index if not)
        if 'what-would' not in pinecone.list_indexes():
            pinecone.create_index('what-would', dimension=len(embeds[0]))
        # connect to index
        index = pinecone.Index('what-would')

        # initialize chunk size to be 40 words

        text_chunked = split_text_into_chunks(context_texts[counter-1], 40)
        batch_size = 32  # process everything in batches of 32, 32 chunks at a time
        for i in tqdm(range(0, len(text_chunked), batch_size)):
            # set end position of batch
            i_end = min(i + batch_size, len(text_chunked))
            # get batch of lines and IDs
            lines_batch = text_chunked[i: i + batch_size]
            ids_batch = [str(n) + '_' + 'Alix Earle' for n in
                     range(i, i_end)]  # labeling with name of person + id for each batch
            # create embeddings
            print('ready to embed')
            res = openai.embeddings.create(model=MODEL, input=lines_batch)
            embeds = [record.embedding for record in res.data]
            # prep metadata and upsert batch
            meta = [{'text': line} for line in lines_batch]
            to_upsert = zip(ids_batch, embeds, meta)
            # upsert to Pinecone
            index.upsert(vectors=list(to_upsert))

add_text_to_db()
print('im out bb')

index = pinecone.Index('what-would')
MODEL = "text-embedding-ada-002"

query = "my boyfriend cheated on me what should i do"
print(query)

xq = openai.embeddings.create(model=MODEL, input=query) #transforms query into embedding
lol = [record.embedding for record in xq.data]
res = index.query([lol], top_k=5, include_metadata=True) #similaritysearch your db with your question, returns 5 chunks in strings

for match in res['matches']: #
    print(f"{match['score']:.2f}: {match['metadata']['text']}")
