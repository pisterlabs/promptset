import chromadb
from chromadb import Settings
from chromadb.api.types import QueryResult
import json
import re
from langchain.embeddings import OpenAIEmbeddings
import time
import core.config as config

# chroma_client = chromadb.HttpClient(host=config.chromadb_uri, port=config.chromadb_port,
#                                     settings=Settings(allow_reset=True, anonymized_telemetry=False))

# chroma's OpenAIEmbeddingFunction limits requests to 3 per min, 
# langchain's OpenAIEmbeddings handles retries.

# persist_directory = "chroma_02"  # local dev path for "script-running"
# persist_directory = "chroma_test"  # local test path for "script-running"
persist_directory = "./core/utils/chroma_02"  # path used in docker
client = chromadb.PersistentClient(path=persist_directory)
embeddings = OpenAIEmbeddings(openai_api_key=config.openai_api_key)
test_collection_name = "collection_name"
cities_collection_name = "getCities"
warehouses_collection_name = "getWarehouses"
query_response_items = ["documents", "distances"]
# query_response_items = ["documents", "distances", "metadatas"]  # with metadata


# heavy, use once, not for in-docker use
def create_embedded_cities_collection(demo=False):
    coll = client.get_or_create_collection(name=cities_collection_name, 
                                           metadata={"hnsw:space": "cosine"},
                                           )
    file_path='../data/' + cities_collection_name + ".json"
    data = json.load(open(file_path, encoding='utf-8-sig'))
    to_be_embedded_list = []

    for item in data:
        description = preprocess_cities_description(item["Description"])
        to_be_embedded_list.append(description)
    
    if demo:
        to_be_embedded_list = to_be_embedded_list[:110]  # demo, first letter 'A' (for cities)
    print(len(to_be_embedded_list))
    # print(to_be_embedded_list[:3])

    embeddings_list = []
    for i in range(0, len(to_be_embedded_list), 5000):
        embeddings_list += embeddings.embed_documents(to_be_embedded_list[i:i+5000])
        time.sleep(30)
    
    print(len(embeddings_list[0]))
    print(len(embeddings_list))

    ids = []
    documents = []
    metadatas = data[:len(embeddings_list)]
    for i in range(len(embeddings_list)):
        ids.append(data[i]['Ref'])
        documents.append(data[i]['Description'])

    coll.add(
        ids=ids,
        embeddings=embeddings_list,
        documents=documents,
        metadatas=metadatas
    )
    print(len(coll.get(include=["documents"])['ids']))


# heavy, use once, not for in-docker use
def create_embedded_warehouses_collection(demo=False):
    coll = client.get_or_create_collection(name=warehouses_collection_name,
                                           metadata={"hnsw:space": "cosine"},
                                           )
    file_path='../data/' + warehouses_collection_name + ".json"
    data = json.load(open(file_path, encoding='utf-8-sig'))
    to_be_embedded_list = []

    for item in data:
        # description = preprocess_warehouses_description(item["Description"])
        short_address = preprocess_warehouses_shortaddress(item["ShortAddress"])
        to_be_embedded_list.append(short_address)
    
    if demo:
        to_be_embedded_list = to_be_embedded_list[:100]  # demo
    print(len(to_be_embedded_list))
    # print(to_be_embedded_list[:3])
    
    embeddings_list = []
    for i in range(0, len(to_be_embedded_list), 5000):
        embeddings_list += embeddings.embed_documents(to_be_embedded_list[i:i+5000])
        time.sleep(30)
    
    print(len(embeddings_list[0]))
    print(len(embeddings_list))

    ids = []
    documents = to_be_embedded_list
    for i in range(len(embeddings_list)):
        ids.append(data[i]['Ref'])
        metadatas = preprocess_warehouses_metadata(data[i])
    
    # test if chroma can handle everything in one go
    coll.add(
        ids=ids,
        embeddings=embeddings_list,  # preprocessed ShortAddress
        # documents = preprocessed ShortAddress, use metadata for ui
        # mb (Description + ", " + CityDescription) for UI
        documents=documents,
        metadatas=metadatas
    )
    print(len(coll.get(include=["documents"])['ids']))


def preprocess_cities_description(description):
    # 'Абазівка (Полтавський р-н..)' -> 'абазівка', search perf
    return description.split('(')[0].strip().lower()


def preprocess_warehouses_description(description):
    pattern1 = r'\s*\(.*?\):'  # Match '(...):' pattern, in our data = '(до 30кг):'
    pattern2 = r'\(ТІЛЬКИ ДЛЯ МЕШКАНЦІВ\)'
    out1 = re.sub(pattern1, '', description)
    out2 = re.sub(pattern2, '', out1)

    return out2.strip().lower()


def preprocess_warehouses_shortaddress(shortAdress):
    pattern = r'\s*\(.*?\):'  # Match '(...):' pattern, in our data = '(до 30кг):'
    s = re.sub(pattern, '', shortAdress).lower()
    city = s.split(',')[0]
    alphanumeric_substrings = find_alphanumeric_substrings(s)
    words = find_ukrainian_words(s)
    words_to_remove = [city, "тільки", "для", "мешканців", "мешк", "магазин", "маг", "вулиця", "вул",
                       "проспект", "просп", "пр", "провулок", "пров", "біля", "відділення", "під'їзд",
                       "п", "корпус", "корп", "в", "клієнтській", "зоні", "від", "секція", "секц", "осбб",
                       "приміщення", "прим", "район", "ран", "р-н", "мікрорайон", "тц", "т-ц", "трц",
                       "площа", "пл", "бульвар", "бульв", "б", "проїзд", "шосе", "будинок", "буд", "с", "м"]
    res = list(set(words) - set(words_to_remove)) + alphanumeric_substrings
    
    return ' '.join(res)


def preprocess_warehouses_metadata(data: json):
    limited_data = {
        "Ref": data["Ref"],
        "Description": data["Description"],
        "Number": data["Number"],
        "CityRef": data["CityRef"],
        "CityDescription": data["CityDescription"],
        "ShortAddress": data["ShortAddress"],
        "TypeOfWarehouse": data["TypeOfWarehouse"],
        "CategoryOfWarehouse": data["CategoryOfWarehouse"],
        "WarehouseIndex": data["WarehouseIndex"],
    }

    return limited_data


def test_preprocess_warehouses_description():
    input_string = "Some ( до 30кг): text(до 30кг): (до 30кг): more text (ТІЛЬКИ ДЛЯ МЕШКАНЦІВ) additional text"
    cleaned = preprocess_warehouses_description(input_string)
    print(cleaned)  # expected - Some text more text  additional text


# used for warehouses querying
def find_alphanumeric_substrings(input_string):
    pattern = r'\b(?:\d+/[а-яА-ЯіІїЇєЄҐґ\d]+|\d+/\d+|\d+[а-яА-ЯіІїЇєЄҐґ]|[\d]+(?!\d)|\
        [а-яА-ЯіІїЇєЄҐґ]+-\d+|\d+[а-яА-ЯіІїЇєЄҐґ]\d+)\b'
    substrings_found = re.findall(pattern, input_string, re.IGNORECASE | re.UNICODE)

    return substrings_found


def test_find_alphanumeric_substrings():
    input_string = "1Текст з2 б1 т-ц Тіпіль-1, 1/3к під'їздом українським цифра 123 144/2, 14б, 3.14, \
    14/22, чоп, тополь-13, and 144,1  1/3ю The values are 144/2бб, 22/3к, 22/к3, тєпіль-1, and 144. 3к1"
    substrings = find_alphanumeric_substrings(input_string)
    print("input = " + input_string)
    print("numbers:" + str(substrings))
    # expected - ['Тіпіль-1', '1/3к', '123', '144/2', '14б', '3', '14', '14/22', 'тополь-13', '144', '1', 
    # '1/3ю', '144/2бб', '22/3к', '22/к3', 'тєпіль-1', '144', '3к1']


# used for warehouses querying
def find_ukrainian_words(input_string):
    pattern = r'\b[а-яА-ЯяІіЇїЄєҐґ]+\b(?:[\'-][а-яА-ЯяІіЇїЄєҐґ]+\b)*'
    #pattern = r'(?:\b[а-яА-ЯіІїЇєЄҐґ]+\b[^\w]*)'
    words_found = re.findall(pattern, input_string, re.IGNORECASE | re.UNICODE)

    return words_found


def test_find_ukrainian_words():
    input_string = "1Текст з2 б1 т-ц Тіпіль-1, 1/3к під'їздом українським цифра 123 144/2, 14б, 3.14, \
    14/22, чоп, тополь-13, and 144,1  1/3ю The values are 144/2бб, 22/3к, 22/к3, тєпіль-1, and 144. 3к1"
    result = find_ukrainian_words(input_string)
    print("words  :" + str(result))
    # expected - ['т-ц', 'Тіпіль', "під'їздом", 'українським', 'цифра', 'чоп', 'тополь', 'тєпіль']


def find_numero_number_or_first_number(input_string):
    pattern = r'№\s*(\d+(?:\d+)?)'
    number_with_numero_sign = re.search(pattern, input_string)
    if number_with_numero_sign:
        return str(number_with_numero_sign.group())

    first_number = re.search(r'\d+', input_string)
    if first_number:
        return str(first_number.group())
    
    return None


def query_cities_collection(query: str, n_results=20) -> QueryResult:
    coll = client.get_collection(name=cities_collection_name)
    query_embedding = embeddings.embed_documents([query.lower()])
    
    results = coll.query(
        # query_texts=[query.lower()],
        query_embeddings=query_embedding,
        n_results=n_results,
        include=query_response_items,
        # where={"metadata_field": "is_equal_to_this"}, # optional filter
        # where_document={"$contains":"search_string"}  # optional filter
    )

    return results


# mb use (Description + ", " + CityDescription) for UI
def query_warehouses_collection(query: str, cityRef=None, n_results=20) -> QueryResult:
    assert query is not None and len(query) > 0, "query must be non-empty string"
    coll = client.get_collection(name=warehouses_collection_name)
    query_lowercase = query.lower()
    query_embedding = embeddings.embed_documents([query_lowercase])
    
    metadata_filter = {}
    if cityRef:
        metadata_filter["CityRef"] = cityRef

    # attempt to find specific branch/postomat via '№123' or just first number in query
    numero_result = None
    numero_number_or_first_number = find_numero_number_or_first_number(query_lowercase)
    if numero_number_or_first_number:
        metadata_filter["Number"] = numero_number_or_first_number
        numero_result = coll.query(
            query_embeddings=query_embedding,
            n_results=1,
            include=query_response_items,
            where=metadata_filter,
        )
        metadata_filter.pop("Number")

    alphanumeric_substrings = find_alphanumeric_substrings(query_lowercase)
    ukrainian_words = find_ukrainian_words(query_lowercase)
    # and:[contains num, contains another num, or:[word1, word2, word3..]]
    # we use: num1 and num2 and (word1 or word2 or word3)
    # coll.query(query_texts="вул", 
    # where_document={
    #   "$and":[
    #       {"$contains":"15"},
    #       {"$contains":"32"},
    #       {"$or":[
    #           {"$contains":"Перемога"},
    #           {"$contains":"Нова"}
    #       ]}
    #   ]}, 
    # include=['documents'])
    alphanumeric_filter = [{"$contains": num} for num in alphanumeric_substrings]
    ukrainian_words_filter = [{"$contains": word} for word in ukrainian_words]
    document_filter = {}
    
    if len(alphanumeric_filter) >= 1 and len(ukrainian_words_filter) >= 1:
        document_filter["$and"] = alphanumeric_filter
        if len(ukrainian_words_filter) > 1:
            document_filter["$and"].append({"$or": ukrainian_words_filter})
        elif len(ukrainian_words_filter) == 1:
            document_filter["$and"].append(ukrainian_words_filter[0])

    if len(alphanumeric_filter) == 0:
        if len(ukrainian_words_filter) > 1:
            document_filter["$or"] = ukrainian_words_filter
        elif len(ukrainian_words_filter) == 1:
            document_filter.update(ukrainian_words_filter[0])

    if len(ukrainian_words_filter) == 0:
        if len(alphanumeric_filter) > 1:
            document_filter["$and"] = alphanumeric_filter
        elif len(alphanumeric_filter) == 1:
            document_filter.update(alphanumeric_filter[0])

    # debug output
    # print(document_filter)
    # print("======================")

    query_results = coll.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=query_response_items,
        where_document=document_filter,
        where=metadata_filter,
    )

    if numero_result:
        for item in query_response_items:
            query_results[item][0] = numero_result[item][0] + query_results[item][0]
    
    return query_results


# used mainly for testing
def get_collection(collection_name):
    coll = client.get_collection(collection_name)
    # arr = [preprocess_warehouses_description(x) for x in 
    #        coll.get(include=["documents"])["documents"][:10]]
    return len(coll.get(include=["documents"])["ids"])


# careful, deletes collection
def delete_collection(collection_name):
    client.delete_collection(name=collection_name)


# create_embedded_cities_collection(demo=False)
# print(query_cities_collection(query="антонівка"))
# print(get_collection(cities_collection_name))

# create_embedded_warehouses_collection(demo=False)
# print(query_warehouses_collection(query="15 32"))
# print(query_warehouses_collection(query="квітнева"))
# print(query_warehouses_collection(query="26249"))
# print(get_collection(warehouses_collection_name))

# careful, deletes collection
# delete_collection(warehouses_collection_name)

# print(persist_directory)
