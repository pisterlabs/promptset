import spacy
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer, util

# CONSTANTS
API_KEY = ""
nlp = spacy.load("en_core_web_sm")  # load the english model for spacy
embeddings_model = OpenAIEmbeddings(openai_api_key=API_KEY)

# questions generated via chatgpt so it might be funky.
user_inputs = [
    # Existing questions:
    "Can you tell me Nokia's current intent score?",
    "The last interaction we had with Yahoo was when?",
    "Identify all the leads linked to Dell.",
    "Which of my accounts have significant whitespace?",
    "Show me the most recent leads from Alibaba.",
    "Where can I find my whitespace data?",
    "What type of leads does Michael possess?",
    "Are there accounts that might have similar victories to Instagram?",
    "Provide the recent successes of HP.",
    "Can you identify every contact related to this account?",
    "Which accounts are next on my list to work on?",
    "How many business agreements are set with Hulu?",
    "Who are the leads Jennifer is currently handling?",
    "Identify the contacts Mike is overseeing.",
    "What's the renewal date for Tidal?",
    "Can you provide details on LG's most recent deal?",
    "Any leads linked to WooCommerce in my list?",
    "Describe the latest event we had with Cisco.",
    "Who was the last to engage with Pinterest on our behalf?",
    "How do Sony's earlier successes appear?",
    "Describe the methodology Kevin uses to boost Instagram engagement.",
    "Provide the newest updates from Jessica's account list.",
    "Who from our leads or contacts attended meetings in the last three days?",
    "Identify every channel partner linked with my portfolio.",
    # New questions:
    "Describe our most recent engagement with Nokia.",
    "Who oversees Yahoo's account on our end?",
    "Are there any prospective deals in the pipeline with Dell?",
    "Has Michael seen success with his current leads?",
    "Compared to the previous month, how is BlackBerry's intent score?",
    "Any notable activities from Alibaba recently?",
    "Identify our main liaison at Tidal.",
    "In what way does Instagram differ in its renewal protocols?",
    "Any upcoming WooCommerce events or webinars we should be aware of?",
    "The last deal Jennifer finalized with HP was when?",
    "Have we gotten any reviews or feedback from Hulu?",
    "Which of Yahoo's contacts recently engaged with our materials?",
    "How are we progressing towards our Q4 objectives with Sony?",
    "What kind of assistance does Pinterest often request?",
    "Did the Cisco team provide comments from our last engagement?",
    "Which accounts managed by Kevin have a promising future?",
    "How many months has Jessica been overseeing LG's account?",
    "Any pending issues or concerns regarding Instagram?",
    "Members of which team frequently engage with Dell?",
    "Outline the next key events or goals for the Nokia account."
]

# List of questions
questions = [
    # Existing questions:
    "What is the intent score of Apple?",
    "When was the last time we interacted with Google?",
    "Who are all the leads associated with Microsoft?",
    "List my accounts with the most whitespace?",
    "What are the last Amazon leads?",
    "Where is my whitespace?",
    "What leads do John have?",
    "Which accounts will result in a win like account Facebook?",
    "What are the last Oracle company wins?",
    "Who are all the contacts associated with this account?",
    "What accounts should I work on next?",
    "How many deals do we have for account Netflix?",
    "What leads are being worked on by Alice?",
    "What contacts are being worked on by Bob?",
    "When is the renewal due for Spotify?",
    "What was the last deal at company Adobe?",
    "Do I have any leads in account Shopify?",
    "What was the last interaction with IBM?",
    "When was the last interaction with account Twitter, and by whom?",
    "What do previous wins look like in Samsung?",
    "How do George generate Facebook pipeline?",
    "What is the latest in Maria's accounts?",
    "Which contacts/leads attended meetings in the past 3 days?",
    "Who are all the channel partners associated with my accounts?",
    # New questions:
    "How did our last interaction with Apple go?",
    "Who is managing the account for Google?",
    "What potential opportunities do we have with Microsoft?",
    "Has John had any success with his leads?",
    "How does Tesla's intent score compare to last month?",
    "Are there any updates on Amazon's recent activity?",
    "Who is our primary contact at Spotify?",
    "How does Facebook's renewal process differ from others?",
    "Are there any upcoming events or webinars for Shopify customers?",
    "When did Emma last close a deal with Oracle?",
    "What feedback have we received from Netflix?",
    "Who were the last contacts from Google to engage with our content?",
    "How are we tracking against our Q4 goals with Samsung?",
    "What type of support does Twitter require?",
    "Is there any feedback from the last meeting with IBM's team?",
    "Which of George's accounts have the highest growth potential?",
    "How long has Maria been managing the Adobe account?",
    "Are there any unresolved issues with Facebook?",
    "Which team members have had the most interaction with Microsoft?",
    "What are the upcoming milestones for the Apple account?"
]


def preprocess(text):
    tokens = [token.text for token in nlp(text)]
    return ' '.join(tokens)


def extract_named_entities(text):
    doc = nlp(text)
    return [(ent.text.lower(), ent.label_) for ent in doc.ents]


def named_entity_embeddings(text):
    named_entities = extract_named_entities(text)
    named_entity_tokens = [f"[{ent_type.upper()}_{ent_text}]" for ent_text, ent_type in named_entities]

    if not named_entity_tokens:
        named_entity_tokens = ["[NO_ENTITY]"]

    entity_embeddings = np.mean([embeddings_model.embed_query(token) for token in named_entity_tokens], axis=0)
    return entity_embeddings


def combine_embeddings(basic_embeddings, entity_embeddings):
    """Combines basic embeddings with named entity embeddings."""
    return [np.concatenate((basic, entity)) for basic, entity in zip(basic_embeddings, entity_embeddings)]


def get_embeddings(texts):
    preprocessed_texts = [preprocess(text) for text in texts]
    basic_embeddings = embeddings_model.embed_documents(preprocessed_texts)
    entity_embeddings_list = [named_entity_embeddings(text) for text in texts]
    return combine_embeddings(basic_embeddings, entity_embeddings_list)


## Todo: remove ensemble attempt here becaues 1- the two models have different sizes that I didn't acount for. 2- I want to compare individual implemenations vs ensemble.
def match_questions(questions, user_input, top_n=3, threshold=0.80):
    """Matches a user input to a list of questions based on similarity."""
    basic_user_input_embed = embeddings_model.embed_query(preprocess(user_input))
    entity_user_input_embed = named_entity_embeddings(user_input)
    user_input_embed_openai = np.concatenate((basic_user_input_embed, entity_user_input_embed))

    question_embeddings = get_embeddings(questions)
    similarities = [util.pytorch_cos_sim(user_input_embed_openai, question_embed)[0][0].item() for question_embed in
                    question_embeddings]

    most_similar_idxs = np.argsort(similarities)[-top_n:][::-1]
    above_threshold_idxs = [idx for idx in most_similar_idxs if similarities[idx] >= threshold]

    results = [(questions[idx], similarities[idx]) for idx in
               most_similar_idxs]  ## instead of a more readable {"question"... like in early attempts, I want it to be available for an abstracted test function taht looks for a pattern like resilt[0][0]

    # For printing/debugging purposes
    if not above_threshold_idxs:
        print("No matches found that meet the strict similarity threshold.")
        above_fallback_threshold_idxs = [idx for idx in most_similar_idxs if similarities[idx] >= threshold - 0.1]
        if above_fallback_threshold_idxs:
            print("Matches above the fallback threshold:")
            for idx in above_fallback_threshold_idxs:
                print(questions[idx], "with similarity score:", similarities[idx])
        else:
            print("No matches found even after considering the fallback threshold.")
    else:
        print("Matches above the threshold:")
        for idx in above_threshold_idxs:
            print(questions[idx], "with similarity score:", similarities[idx])

    print("\nTop 3 results for debugging:")
    for res in results:
        print(res[0], "with similarity score:", res[1])

    return results

# Uncomment for testing
# match_questions(questions, "what are my best accounts")
