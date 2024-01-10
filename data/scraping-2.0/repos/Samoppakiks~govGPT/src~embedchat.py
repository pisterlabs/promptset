# from .config import openaiapi, openaiorg, pinecone_api_key, pinecone_environment
import os
import openai
import pinecone
import pickle
import re
import re
import json
import time
from spacy.lang.en import English


openaiapi = os.environ.get("OPENAI_API_KEY")
openaiorg = os.environ.get("OPENAI_ORG_ID")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_environment = os.environ.get("PINECONE_ENV")

openai.api_key = openaiapi
openai.organization = openaiorg

nlp = English()  # Just the language with no model
nlp.add_pipe("sentencizer")  # Adding a sentencizer pipeline component


def split_sentences(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]


def split_into_chunks(text, max_len=800):
    sentences = split_sentences(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_len:
            current_chunk += sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks


def clean_and_split_text(text):
    # Remove extra newline characters and join the text
    text = " ".join(text.strip().split("\n"))
    # Remove page numbers
    text = re.sub(r"\d+\n", "", text)
    # Remove citations
    # text = re.sub(r"(?:\*\s*[A-Za-z\d*]+\s*vide[^“]*?(?:\n|$))", "", text)
    # Identify rule titles and add a separator before them
    text = re.sub(r"(\d+(\.|\')\.?\s[^—]+[,—])", r"@@@\1", text)
    # Split the text based on the separator
    segments = text.split("@@@")
    # Create a list to store the cleaned segments
    cleaned_segments = []
    for segment in segments:
        # Only remove extra spaces and newline characters
        segment = re.sub(r"\s+", " ", segment).strip()

        if len(segment) > 800:
            split_chunks = split_into_chunks(segment)
            cleaned_segments.extend(split_chunks)
        else:
            cleaned_segments.append(segment)
    cleaned_segments = [segment for segment in cleaned_segments if segment.strip()]
    return cleaned_segments


def write_chunks_to_file(chunks, pdf_path, namespace=None):
    # Create a 'chunks' directory if it doesn't exist
    if not os.path.exists("chunks"):
        os.makedirs("chunks")
    # Set the output file name using the original PDF filename
    if pdf_path:
        output_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    else:
        output_filename = namespace
    output_file_path = f"./chunks/{output_filename}_chunks.txt"
    # Write the chunks to the output file
    with open(output_file_path, "w") as f:
        for idx, chunk in enumerate(chunks, start=1):
            f.write(f"Chunk {idx}:\n")
            f.write(chunk)
            f.write("\n\n")


def process_extracted_text(
    query,
    text,
    pdf_path,
    search_scope="current_file",
    namespace=None,
    department=None,
    type_of_document=None,
    year=None,
):
    # selecting the huggingface tokeniser and selecting the chunk sizes

    texts = []
    # max_length = 4000
    # overlap = 100

    # splitting the text into chunks using our custom function
    texts = clean_and_split_text(text)
    write_chunks_to_file(texts, pdf_path, namespace)

    # initialising the openai api key
    model_engine = "text-embedding-ada-002"

    # initialising pinecone
    pinecone.init(
        api_key=pinecone_api_key,
        environment=pinecone_environment,
    )

    # fetching the name of the created index and initialising it
    index_name = "rajgov"
    index = pinecone.Index(index_name)

    # creating embeddings of chunks and uploading them into the index
    # Get embeddings for the PDF file
    if pdf_path:
        file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    else:
        file_name = namespace
    embeddings_file_path = f"./embeddings/{file_name}_embeddings.pkl"

    if namespace is None:
        namespace = file_name

    embeddings = []
    if not os.path.exists(embeddings_file_path):
        # creating embeddings of chunks and save them to a file
        for i, chunk in enumerate(texts):
            response = openai.Embedding.create(input=[chunk], model=model_engine)
            embedding = response["data"][0]["embedding"]
            metadata = {"text": chunk}
            if department is not None:
                metadata["department"] = department
            if type_of_document is not None:
                metadata["type_of_document"] = type_of_document
            if year is not None:
                metadata["year"] = year
            embeddings.append((f"chunk_{i}", embedding, metadata))

            with open(embeddings_file_path, "ab") as f:
                print(f"Saving embeddings of chunk_{i} to {embeddings_file_path}")
                pickle.dump([(f"chunk_{i}", embedding, metadata)], f)

            # Upserting embeddings to namespace
            index.upsert(
                vectors=[(f"chunk_{i}", embedding, metadata)], namespace=namespace
            )
    else:
        # load embeddings from the file
        with open(embeddings_file_path, "rb") as f:
            print(f"Loading embeddings from {embeddings_file_path}")
            while True:
                try:
                    embeddings.append(pickle.load(f))
                except EOFError:
                    break

        completed_chunks = len(embeddings)
        print(f"Completed chunks: {completed_chunks}")

        # Continue creating embeddings from where it left off
        for i, chunk in enumerate(texts[completed_chunks:], start=completed_chunks):
            response = openai.Embedding.create(input=[chunk], model=model_engine)
            embedding = response["data"][0]["embedding"]
            metadata = {"text": chunk}
            if department is not None:
                metadata["department"] = department
            if type_of_document is not None:
                metadata["type_of_document"] = type_of_document
            if year is not None:
                metadata["year"] = year
            embeddings.append((f"chunk_{i}", embedding, metadata))

            with open(embeddings_file_path, "ab") as f:
                print(f"Saving embeddings of chunk_{i} to {embeddings_file_path}")
                pickle.dump([(f"chunk_{i}", embedding, metadata)], f)

            # Upserting embeddings to namespace
            index.upsert(
                vectors=[(f"chunk_{i}", embedding, metadata)], namespace=namespace
            )

    # preparing the query
    """query = translate_to_english_chatgpt(query)
    focus_phrases = extract_focus_phrases(query)
    print(f"QUERY: {query}")"""

    # querying the index
    query_response = openai.Embedding.create(input=[query], model=model_engine)
    query_embedding = query_response["data"][0]["embedding"]

    # the response will be in json with id, metadata with text, and score
    if search_scope == "current_file":
        results = index.query(
            queries=[query_embedding],
            top_k=5,
            include_metadata=True,
            namespace=namespace,
        )
    else:  # search_scope == 'entire_database'
        results = index.query(queries=[query_embedding], top_k=5, include_metadata=True)
    print(results)

    answer, search_results = chatgpt_summarize_results(
        query, results
    )  # focus_phrases,)

    print(f"ANSWER: {answer}")

    return answer, search_results


def chatgpt_summarize_results(query, results):  # focus_phrases)
    search_results = ""
    for match in results["results"][0]["matches"]:
        score = match["score"]
        text = match["metadata"]["text"]
        search_results += f"{score:.2f}: {text}\n"
    print(search_results)

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant working at a government help center facilities. People ask you questions related to permissable activities,and  for information on government services.",
            },
            {
                "role": "user",
                "content": f"The query is: '{query}'. Based on the following search results, provide an answer to the query, after considering each result with respect to the query and checking if anything related to the query can be inferred from each result. Finally, comment on reason for your final interpreation, as well as any additional information that may not be contained in the text that may help answer the query. considering not only exact matches but also possible inferences about the expected action that can be made based on the results. :\n\n{search_results}",  # You may also use the focus phrases : {focus_phrases} for better inference.
            },
        ],
    )

    gpt_response = response.choices[0].message["content"].strip()

    return gpt_response, search_results


def chatgpt_get_response(context, query):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant working at a government help center facilities. People ask you questions related to permissable activities, and for information on government services.",
            },
            {"role": "user", "content": context},
            {"role": "user", "content": query},
        ],
    )

    return response.choices[0].message["content"].strip()


# just as an aside, the following prompts gave 2 different results when run twice, without making any change to
# prompt or query. One said permission can be given, the other said permission cannot be given.
# here are the prompts
""" for all 3 : role : system : You are a helpful assistant working at a government help center facilities. People ask you questions related to permissable activities,and for information on government services
for 1 : role : user : Translate the following text to English: Then give answers on  what is the context of the text and what is the action expected. Finally , convert the text into an appropriate query that can be used to search through a semantic database to search through embedded text to find the right text that can give an answer on whether the action expected can be peformed within the given context. Make sure that the query is appropriate for searching through a semantic database consisting of government laws and regulations, and avoid adding those phrases to the query that may unnecessarily confuse the search engine.'
for 2 : role : user : Extract key phrases from the following query: Only extract the phrases related to actions that are expected (including giving permissions under existing regulations, asking for details regulations relevant to the case etc), that can be searched through a datatbase of governemnt acts and regulations. Avoid extracting phrases that are not relevant to the semantic search through a database of government rules.
for 3 : role : user : The query is: Based on the following search results, provide an answer to the query, after considering each result with respect to the query and checking if anything related to the query can be inferred from each result. Finally, comment on reason for your final interpreation, as well as any additional information that may not be contained in the text that may help answer the query. considering not only exact matches but also possible inferences about the expected action that can be made based on the results. You may also use the focus phrases :  for better inference.:"""


"""def split_text(text, max_chunk_size=300, overlap=50):
    pattern = r'(\d+(\.\d+)*\s*\w*\.?)'
    chunks = re.split(pattern, text)

    combined_chunks = []
    for i in range(0, len(chunks)-1, 2):
        chunk = ''
        if chunks[i]:
            chunk += chunks[i]
        if chunks[i+1]:
            chunk += chunks[i+1]
        combined_chunks.append(chunk)

    cleaned_chunks = [chunk.replace('\n', ' ').strip()
                      for chunk in combined_chunks]

    smaller_chunks = []
    for chunk in cleaned_chunks:
        rule_number_match = re.match(r'(\d+(\.\d+)*)', chunk)
        if rule_number_match:
            rule_number = rule_number_match.group(1)
            chunk = re.sub(r'\d+(\.\d+)*', '', chunk).strip()
        else:
            rule_number = ''

        tokens = chunk.split()

        for i in range(0, len(tokens), max_chunk_size - overlap):
            start = max(0, i - overlap) if i > 0 else i
            end = min(i + max_chunk_size, len(tokens))
            small_chunk = ' '.join(tokens[start:end])
            if rule_number:
                if start > 0:
                    small_chunk = f"{rule_number} (cont'd) " + small_chunk
                else:
                    small_chunk = f"{rule_number} " + small_chunk
            smaller_chunks.append(small_chunk)
    total_chunks = len(smaller_chunks)
    print(f"Total number of chunks created: {total_chunks}")
    return smaller_chunks


Previous answer :
with press or radio—Previous sanction of the Government shall not be required when the member of the service, in the bonafide discharge of his duties or otherwise, publishes a book or contributes to or participates in a public media. Provided that he shall observe the provisions of rules and at all times make it clear that the views expressed, are of his own and not those of the Government. 7. Criticism
0.84: should not be given to a moS to publish articles in the journals, souvenirs etc., of political parties: - A doubt has been raised whether members of the All India Services can be permitted to publish articles in the journals, souvenirs etc., of political parties. 2. The
0.81: 1995) 185 Provided that nothing in this rule shall apply to any statement made or views expressed by a member of the Service in his official capacity and in the due performance of the duties assigned to him. (GOI Instructions: D.P. & A.R. letter No. 11017/9/75—AlS(III), dated the 2nd March, 1976, reproduced under Miscellaneous Executive Instructions at the end of these Rules) 8. Evidence
0.81: Government may, however, at any time disallow the officer from pursuing his application for admission of financial assistance in cases where Govt. is of the view that 207 a member of the service has utilized his official influence to obtain the same or if the Government feels that such course of action is not in the interest of the Government. [Letter No. 11017/18/91-AIS(III)dated 1.7.
0.81: from literary, cultural or artistic efforts, which are not aided by the knowledge acquired by the member in the course of his service, is not ‘fee’ for the purpose of SR 12 and can be retained by the officer in full: - A question arose, whether a member of the service could accept royalty of the publication of a book of literary, artistic, or scientific character and also whether such royalties were to be treated as ‘Fee’ u
Previous query: 
Can an officer publish an article in a journal?
Previous final answer: 
Based on the search results, it can be inferred that a member of the service is allowed to publish a book or contribute to a public media in the bonafide discharge of his duties or otherwise, without requiring the previous sanction of the government. However, it is mandatory for the member to observe the provisions of rules and ensure that the views expressed are his own and not those of the government. It is not clear whether the query refers to a civil or police officer, but it does not seem to be prohibited unless it is a publication in a journal of a political party. It is important to note that any statements made or views expressed in the due performance of the duties assigned to him by a member of the service in his official capacity exempt him from the rule."""


"""def translate_to_english_chatgpt(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant working at a government help center facilities. People ask you questions related to permissable activities,and for information on government services."},
            {"role": "user", "content": f"Translate the following text to English: '{text}'.Then give answers on  what is the context of the text and what is the action expected. Finally , convert the text into an appropriate query that can be used to search through a semantic database to search through embedded text to find the right text that can give an answer on whether the action expected can be peformed within the given context. Make sure that the query is appropriate for searching through a semantic database consisting of government, regulations,polcies, programmes and other government services, and avoid adding those phrases to the query that may unnecessarily confuse the search engine."}
        ]
    )
    translated_text = response.choices[0].message['content'].strip()
    print(f"Translated text : '{translated_text}")
    return translated_text


def extract_focus_phrases(translated_text):

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant working at a government help center facilities. People ask you questions related to permissable activities,and for information on government services.."},
            {"role": "user",
                "content": f"Extract key phrases from the following query: '{translated_text}'. Only extract the phrases related to actions that are expected (including giving permissions under existing regulations, asking for details within government schemes, looking for legal advice etc), that can be searched through a datatbase of governemnt acts, regulations, policies and welfare schemes. Avoid extracting phrases that are not relevant to the semantic search through such a database."}
        ]
    )
    focus_phrases = response.choices[0].message['content'].strip()
    print(focus_phrases)
    print(f"Focus phrases : '{focus_phrases}")
    return focus_phrases"""
