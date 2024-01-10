import sys
from googleapiclient.discovery import build
import requests
from bs4 import BeautifulSoup
import spacy
from spanbert import SpanBERT
from spacy_help_functions import extract_relations
from spacy_help_functions import create_entity_pairs
from collections import defaultdict
import openai
import time

# Global Variables
EXTRACTION_METHOD = ""
GOOGLE_API_KEY = ""
GOOGLE_ENGINE_ID = ""
OPENAI_SECRET_KEY = ""
r = 0
t = 0
q = ""
k = 0
# four types of relations
RELATION = ["Schools_Attended", "Work_For", "Live_In", "Top_Member_Employees"]
# internal name
INTERNAL_NAME = ["per:schools_attended", "per:employee_of", "per:cities_of_residence", "org:top_members/employees"]
# named entities
ENTITY = [["PERSON", "ORGANIZATION"], ["PERSON", "ORGANIZATION"], ["PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"], ["ORGANIZATION", "PERSON"]]
# gpt output example
OUTPUT_EXAMPLE = ['["Jeff Bezos", "Schools_Attended", "Princeton University"]', '["Alec Radford", "Work_For", "OpenAI"]', '["Mariah Carey", "Live_In", "New York City"]', '["Jensen Huang", "Top_Member_Employees", "Nvidia"]']


def get_parameters():
    """
    Get parameters from the inputs, and store the value in global variables.
    """
    inputs = sys.argv
    global EXTRACTION_METHOD, GOOGLE_API_KEY, GOOGLE_ENGINE_ID, OPENAI_SECRET_KEY, r, t, q, k
    # If the input format or value is incorrect, print the error message
    if len(sys.argv) != 9:
        print("Correct format: python3 project2.py [-spanbert|-gpt3] <google api key> <google engine id> <openai secret key> <r> <t> <q> <k>")
        exit(0)
    elif int(sys.argv[5]) not in [1, 2, 3, 4]:
        print("The range of <r> is 1 - 4")
        exit(0)
    elif float(sys.argv[6]) < 0 or float(sys.argv[6]) > 1:
        print("The range of <t> is 0 - 1")
        exit(0)
    elif int(sys.argv[8]) <= 0:
        print("The range of <k> is 1 - INF")
        exit(0)
    else:
        EXTRACTION_METHOD = inputs[1]
        GOOGLE_API_KEY = inputs[2]
        GOOGLE_ENGINE_ID = inputs[3]
        OPENAI_SECRET_KEY = inputs[4]
        r = int(inputs[5])
        t = float(inputs[6])
        q = inputs[7]
        k = int(inputs[8])
        # Print parameters
        print("Parameters: ")
        print("Client key  = " + GOOGLE_API_KEY)
        print("Engine key  = " + GOOGLE_ENGINE_ID)
        print("OpenAI key  = " + OPENAI_SECRET_KEY)
        print("Method      = " + EXTRACTION_METHOD[1:])
        print("Relation    = " + RELATION[r-1])
        print("Threshold   = " + str(t))
        print("Query       = " + q)
        print("# of Tuples = " + str(k))
        print("Loading necessary libraries; This should take a minute or so ...)")

def google_search(**kwargs):
    """
    Get the result from Google Search Engine.
    """
    service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
    res = service.cse().list(q=q, cx=GOOGLE_ENGINE_ID, **kwargs).execute()
    result = res['items']
    return result

def extract_plain_text(url):
    """
    Retrieve the corresponding webpage; if we cannot retrieve the webpage (e.g., because of a timeout), just skip it and move on.
    Extract the actual plain text from the webpage using Beautiful Soup.
    """
    # reference: https://stackoverflow.com/questions/1936466/how-to-scrape-only-visible-webpage-text-with-beautifulsoup/24968429#24968429
    try:
        text = requests.get(url, timeout=5).text
        soup = BeautifulSoup(text, 'html.parser')
        # kill all unwanted elements
        unwanted = ['style', 'script', 'head', 'title', 'meta', '[document]']
        for s in soup(unwanted):
            s.extract()
        # get text
        text = soup.get_text(separator="\n")
        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # drop blank lines
        text = ' '.join(chunk for chunk in chunks if chunk)
        # If the resulting plain text is longer than 10,000 characters, truncate the text to its first 10,000 characters (for efficiency) and discard the rest.
        if len(text) > 10000:
            print(f"\tTrimming webpage content from {len(text)} to 10000 characters")
            text = text[:10000]
        # print(text)
        print(f"\tWebpage length (num characters): {len(text)}")
    except Exception as e:
        print(e)
        return
    return text

def use_spanbert():
    global q
    iteration = -1
    # record already-seen URLs
    previous_url = set()
    # record used query
    previous_query = set()
    previous_query.add(q)
    # Apply spacy model to raw text (to split to sentences, tokenize, extract entities etc.)
    nlp = spacy.load("en_core_web_lg")
    # Load pre-trained SpanBERT model
    spanbert = SpanBERT("./pretrained_spanbert") 
    # record tuples
    X = defaultdict(int)
    while True:
        iteration += 1
        print(f"=========== Iteration: {iteration} - Query: {q} ===========\n")
        # Query Google Custom Search Engine to obtain the URLs for the top-10 webpages for query q
        results = google_search(num=10)
        # For each URL from the previous step that has not processed, extract tuples
        num_url = 0
        for result in results:
            num_url += 1
            url = result['link']
            print(f"URL({num_url} / 10): {url}")
            # skip already-seen URLs
            if url in previous_url:
                print(f"\tSkip the already-seen URL")
                continue
            previous_url.add(url)

            print("\tFetching text from url ...")
            # Extract the actual plain text from the webpage using Beautiful Soup.
            plain_text = extract_plain_text(url)

            print("\tAnnotating the webpage using spacy...")
            # Use the spaCy library to split the text into sentences and extract named entities
            doc = nlp(plain_text)
            # use the sentences and named entity pairs as input to SpanBERT to predict the corresponding relations, and extract all instances of the relation specified by input parameter r.
            extract_relations(X, doc, spanbert, ENTITY[r-1], INTERNAL_NAME[r-1], t)
        
        # if len(X) >= k:
        #     print(f"================== ALL RELATIONS for {INTERNAL_NAME[r-1]} ( {k} ) =================")
        # else:
        #     print(f"================== ALL RELATIONS for {INTERNAL_NAME[r-1]} ( {len(X)} ) =================")

        print(f"================== ALL RELATIONS for {INTERNAL_NAME[r-1]} ( {len(X)} ) =================")
        #  sort tuples in decreasing order by extraction confidence
        sorted_X = sorted(X.items(), key=lambda item: item[1], reverse=True)
        # num_relation = 0
        for key, value in sorted_X:
            print(f"Confidence: {value}          | Subject: {key[0]}               | Object: {key[2]}")
            # num_relation += 1
            # if num_relation >= k:
            #     break
        # If X contains at least k tuples, return
        if len(sorted_X) >= k:
            break
        # select from X a tuple y has not been used for querying yet and has an extraction confidence that is highest among the tuples in X
        else:
            flag = False
            for key, value in sorted_X:
                query = f"{key[0]} {key[2]}"
                if query not in previous_query:
                    q = query
                    previous_query.add(query)
                    flag = True
                    break
            # If no such y tuple exists, then stop.
            if not flag:
                print (f"ISE has stalled before retrieving {k} high-confidence tuples.")
                break
    
    print(f"Total # of iterations = {iteration + 1}")
    return

def get_openai_completion(prompt, model, max_tokens, temperature, top_p, frequency_penalty, presence_penalty):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )
    response_text = response['choices'][0]['text']
    return response_text

def extract_relations_gpt3(X, doc, entities_of_interest, internal_name):
    """
    This function is a rewrite of the ``extract_relations()`` function in ``spacy_help_functions.py`` when it is used on gpt3.
    """

    num_sentences = len([s for s in doc.sents])
    print("\tExtracted {} sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...".format(num_sentences))

    cur_sentences = -1
    extracted_sentences = 0
    extracted_relations = 0
    overall_relations = 0

    for sentence in doc.sents:
        flag = False
        cur_sentences += 1
        if cur_sentences % 5 == 0:
            print("\tProcessed {} / {} sentences".format(cur_sentences, num_sentences))

        entity_pairs = create_entity_pairs(sentence, entities_of_interest)
        examples = []
        for ep in entity_pairs:
            # not run gpt3 over entity pairs that do not contain named entities of the right type for the relation of interest r.
            if ep[1][1] == entities_of_interest[0] and ep[2][1] in entities_of_interest[1:]:
                examples.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})
            elif ep[2][1] == entities_of_interest[0] and ep[1][1] in entities_of_interest[1:]:
                examples.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})
        if not examples:
            continue
        # gpt prompt
        prompt = """ Given a sentence, extract all instances of the following relationship types as possible:
        relationship type: {}
        Output: ["SUBJECT ENTITY", "RELATIONSHIP", "OBJECT ENTITY"]
        Sample Output: {}
        Sentence: {} 
        Output: """.format(RELATION[r-1], OUTPUT_EXAMPLE[r-1], sentence)

        model = 'text-davinci-003'
        max_tokens = 100
        temperature = 0.2
        top_p = 1
        frequency_penalty = 0
        presence_penalty = 0

        relations = get_openai_completion(prompt, model, max_tokens, temperature, top_p, frequency_penalty, presence_penalty)
        time.sleep(1.5)

        relations = relations.split(']')
        # print(relations)
        for relation in relations:
            # relation = relation.split(',')
            relation = relation.split('"')
            # print(relation)
            # print(len(relation))
            if len(relation) < 7:
                continue
            subj = relation[1]
            obj = relation[5]
            # print(relation[0][3:-1])
            # print(relation[1][2:-1])
            # print(relation[2][2:-1])
            if len(subj.strip()) <= 0 or len(obj.strip()) <= 0:
                continue
            if not flag:
                extracted_sentences += 1
                flag = True
            overall_relations += 1
            print("\n\t\t=== Extracted Relation ===")
            print("\t\tSentence: {}".format(sentence))
            print("\t\tSubject: {} ; Object: {} ;".format(subj, obj))
            if (subj, obj) in X:
                print("\t\tDuplicate. Ignoring this")
            else:
                X[(subj, obj)] = 1
                extracted_relations += 1
                print("\t\tAdding to set of extracted relations")
            print("\t\t==========")

    print("\n\tExtracted annotations for  {}  out of total  {}  sentences".format(extracted_sentences, num_sentences))
    print("\tRelations extracted from this website: {} (Overall: {})\n".format(extracted_relations, overall_relations))

    return

def use_gpt3():
    global q
    iteration = -1
    # record already-seen URLs
    previous_url = set()
    # record used query
    previous_query = set()
    previous_query.add(q)
    # Apply spacy model to raw text (to split to sentences, tokenize, extract entities etc.)
    nlp = spacy.load("en_core_web_lg")
    # openAI key
    openai.api_key = OPENAI_SECRET_KEY
    # record tuples
    X = defaultdict(int)
    while True:
        iteration += 1
        print(f"=========== Iteration: {iteration} - Query: {q} ===========\n")
        # Query Google Custom Search Engine to obtain the URLs for the top-10 webpages for query q
        results = google_search(num=10)
        # For each URL from the previous step that has not processed, extract tuples
        num_url = 0
        for result in results:
            num_url += 1
            url = result['link']
            print(f"URL({num_url} / 10): {url}")
            # skip already-seen URLs
            if url in previous_url:
                print(f"\tSkip the already-seen URL")
                continue
            previous_url.add(url)

            print("\tFetching text from url ...")
            # Extract the actual plain text from the webpage using Beautiful Soup.
            plain_text = extract_plain_text(url)

            print("\tAnnotating the webpage using spacy...")
            # Use the spaCy library to split the text into sentences and extract named entities
            doc = nlp(plain_text)
            # uses the OpenAI GPT-3's API for relation extraction.
            extract_relations_gpt3(X, doc, ENTITY[r-1], INTERNAL_NAME[r-1])
        
        # if len(X) >= k:
        #     print(f"================== ALL RELATIONS for {INTERNAL_NAME[r-1]} ( {k} ) =================")
        # else:
        #     print(f"================== ALL RELATIONS for {INTERNAL_NAME[r-1]} ( {len(X)} ) =================")

        print(f"================== ALL RELATIONS for {RELATION[r-1]} ( {len(X)} ) =================")

        # num_relation = 0
        for subj, obj in X.keys():
            print(f"Subject: {subj}             | Object: {obj}")
            # num_relation += 1
            # if num_relation >= k:
            #     break
        # If X contains at least k tuples, return
        if len(X) >= k:
            break
        # select from X a tuple y has not been used for querying yet.
        else:
            flag = False
            for subj, obj in X.keys():
                query = f"{subj} {obj}"
                if query not in previous_query:
                    q = query
                    previous_query.add(query)
                    flag = True
                    break
            # If no such y tuple exists, then stop.
            if not flag:
                print (f"ISE has stalled before retrieving {k} high-confidence tuples.")
                break
    
    print(f"Total # of iterations = {iteration + 1}")
    return


def main():
    # Get the value of the input parameters.
    get_parameters()
    # -spanbert method
    if EXTRACTION_METHOD == "-spanbert":
        use_spanbert()
        return
    # -gpt3 method
    elif EXTRACTION_METHOD == "-gpt3":
        use_gpt3()
        return
    else:
        print("Wrong extraction method")
        return

if __name__ == '__main__':
    main()
