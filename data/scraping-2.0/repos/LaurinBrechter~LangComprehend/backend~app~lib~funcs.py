import json
from langchain.document_loaders import YoutubeLoader
import tiktoken
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
from lib.data_structs import (
    Text, 
    VocabAnswer,
)
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI



def cut_text(text, frac):
    splitted_text = text.split()
    n_words = len(splitted_text)
    # print(n_words)
    lim = int(frac*n_words)
    text_red = splitted_text[:lim]
    return " ".join(text_red), n_words


def get_video_text(url:str, language) -> None | str:
    loader = YoutubeLoader.from_youtube_url(url, language=language)
    result = loader.load()

    if result == []:
        return None

    return result[0].page_content


def get_n_tokens(text) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def get_response_chat(language, text):

    messages = [
        SystemMessage(content=f"""You are a helpful assistant that only provides answers in {language}"""),
        HumanMessage(content=text),
    ]

    return messages

def correct_vocab(vocab_solution:VocabAnswer) -> str:
    answer = vocab_solution.user_translation
    original_text = vocab_solution.original_text

    load_dotenv()
    model = ChatOpenAI(temperature=0)

    return model.predict(
        f"""
        You will be given an original text and its translation. Please state whether the translation is correct or not.

        Original text: 
        {original_text}

        Translation:
        {answer}

        you MUST answer with 'True' if the translation correct and 'False' otherwiese.
        """
    )



def split_into_paras(text, nlp, num_paragraphs=3):

    doc = nlp(text)
    sents = list(doc.sents)
    paras = []
    step_size = len(sents)//num_paragraphs

    for idx in range(0, len(sents), step_size+1):
        paras.append([i.text for i in sents[idx:idx+step_size+1]])

    return [' '.join(i) for i in paras]



def get_qa_topic(num_questions, text:Text, language:str, nlp, dummy=True) -> dict:

    if dummy:
        return """{
            "questions": [
                "Quels sont les trois enjeux majeurs des tensions entre Taïwan et la Chine ?",
                "Quels sont les obstacles potentiels à une invasion de Taïwan par la Chine ?",
                "Quelles seraient les conséquences d'un conflit entre Taïwan et la Chine ?"
            ],
            "answers": [
                "Les trois enjeux majeurs des tensions entre Taïwan et la Chine sont historiques, politiques et stratégiques.",
                "Les obstacles potentiels à une invasion de Taïwan par la Chine sont les montagnes escarpées à l'est de l'île, les infrastructures solides de Taïwan et le large détroit avec des eaux agitées.",
                "Les conséquences d'un conflit entre Taïwan et la Chine seraient lourdes, notamment un emballement régional, un blocus économique avec des conséquences pour les deux pays et le reste du monde, et des perturbations dans le commerce maritime international."
            ],
            "topics": [
                "Tensions entre Taïwan et la Chine",
                "Obstacles potentiels à une invasion de Taïwan",
                "Conséquences d'un conflit entre Taïwan et la Chine"
            ]
            }
            """

    paras = split_into_paras(text.text, nlp[language], num_paragraphs=num_questions)

    load_dotenv()
    model = ChatOpenAI(temperature=0)
    qts = {"questions":[], "topics":[], "chunks":[]}

    for para in paras:

        inp = f"""
            Please come up with a comprehension question in {language} and a topic about the following paragraph:
            ----------------
            {para}
            ----------------
            Output your response as a json with the keys 'question' and 'topic'.
        """

        qt = json.loads(model.predict(inp))

        qts["questions"].append(qt["question"])
        qts["topics"].append(qt["topic"])
        qts["chunks"].append(para)

    return qts


                
    
def get_vocab(pipeline, text:str, irrel:list[str]) -> dict:
    doc = pipeline(text)
    
    voc = {}
    ents = doc.ents
    irrel = ["PUNCT", "SPACE", "NUM"]

    for token in doc:
        tok_str = str(token).lower()
        lemma = token.lemma_.lower()
        if token.pos_ not in irrel:
            if lemma in voc.keys():
                if tok_str not in voc[lemma]:
                    voc[lemma].append(tok_str)
            else:
                voc[lemma] = [tok_str]
    return {"vocab":voc, "entities":ents}


def prompt_sentence(words:list[str], language:str) -> str:
    model = ChatOpenAI(temperature=0)
    
    prompt = f"""Please generate a {language} sentence with a conjugated 
    form of the word ```{', '.join(words)}``` in it."""
    print(prompt)
    return model.predict(prompt)

def translate(sentence:str, from_:str, to:str) -> str:
    model = ChatOpenAI(temperature=0)

    prompt = f"""Please translate the following sentence 
    from {from_} to {to}:
    {sentence}    
    """

    return model.predict(prompt)
