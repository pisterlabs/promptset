import json
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from sqlitedict import SqliteDict
db = SqliteDict('./db.sqlite', autocommit=True)
#from llama_cpp.llama import Llama, LlamaGrammar
import httpx
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 800,
    chunk_overlap  = 60,
    length_function = len,
    add_start_index = True,
)

def extract_keywords(tekst, llm_chain):
    texts = text_splitter.create_documents([tekst])
    responses = []
    for doc in texts:
        response = llm_chain.run(doc.page_content)
        responses.append(response)
    return responses





template = """<s>[INST]<<SYS>> Wymień słowa kluczowe z dokumentu, oddzielone przecinkami: <</SYS>>
Dokument:  Szanowna Pani Marszałek! Abraham Lincoln miał takie powiedzenie: Czasem możesz kogoś oszukać, ale nie możesz okłamywać wszystkich cały czas. Premier Morawiecki konsekwentnie stara się tę mądrość obalić. Nic dziwnego, że pana wystąpienie zostało nazwane exposé kłamstw. I pozostał symbol tego wystąpienia - prezes Marian Banaś tam siedzący, oklaskujący na stojąco premiera. Każdego dnia pojawiają się nowe informacje, jak jego współpracownicy okradali, oszukując na VAT, polskich emerytów, pacjentów i niepełnosprawnych. Premier Morawiecki może mówić, że stworzył dobrobyt i Polacy na Wyspach pakują już walizki, żeby tu wrócić. A jaka jest, Wysoka Izbo, sytuacja? W 2018 r. zmarło 414 tys. osób, najwięcej od II wojny światowej, zapaść służby zdrowia, armagedon na SOR i rekordowy poziom skrajnego ubóstwa. Żyje w nim 5,4%  Polaków. Wstyd, panie premierze, za te kłamstwa.[/INST]
Słowa kluczowe: służba zdrowia, ubóstwo, kłamstwa, exposé, afery </s>
<s>[INST]Dokument: Jacka Rostowskiego ze słynnym: na te obietnice, które składa Prawo i Sprawiedliwość, pieniędzy nie ma i w ciągu 4 najbliższych lat nie będzie, czy też samego Donalda Tuska: jeżeli ktoś wie, gdzie leżą zakopane w Polsce miliardy, które można porozdawać ludziom, to nie powinien z tym zwlekać.Z tego miejsca odpowiem panu premierowi Tuskowi. Tymi osobami, które wiedziały, gdzie nie są zakopane, ale ukradzione przez mafie VAT-owskie pieniądze, byli pan prezes Jarosław Kaczyński oraz pan premier Mateusz Morawiecki.Również minister Banaś, tak jest.Ale oczywiście o sukcesach polskiej gospodarki świadczy nie tylko wzrost przychodów budżetowych. Wszak do woli możemy żonglować wskaźnikami finansowymi i gospodarczymi. Bezrobocie z poziomu 8% w 2015 r. zjechało do 3,3% według najnowszych danych[/INST]
Słowa kluczowe: gospodarka, finanse, bezrobocie, mafia VAT-owska </s>
<s>[INST]Dokument: {question} [/INST]
Słowa kluczowe:
"""

#grammar_text = httpx.get("https://raw.githubusercontent.com/ggerganov/llama.cpp/master/grammars/list.gbnf").text

prompt = PromptTemplate(template=template, input_variables=["question"])
#grammar = LlamaGrammar.from_string(grammar_text)

#LlamaCpp.update_forward_refs()


llm = LlamaCpp(
    model_path="../models/trurl-2-13b-instruct-q4_K_M.gguf",  
    verbose=True,
    temperature=0.5,
    n_ctx=4096,
    n_gpu_layers=30,
    mlock = True,
    stop = ['</s>'],
)


llm_chain = LLMChain(prompt=prompt, llm=llm)

#for i in range(110, 111):

#get list of files in directory
files = os.listdir(r"C:\Users\pawel\Development\Sejm_Scraper")
#filter files which begin with posiedzenie and are json files
files = [file for file in files if file.startswith("posiedzenie") and file.endswith(".json")]

for i in range(1,len(files)+1):

    with open(f"C:\\Users\\pawel\\Development\\Sejm_Scraper\\posiedzenie_{i}.json", 'r', encoding = 'utf8') as fin:
        posiedzenie = json.load(fin)
    db_posiedzenie = SqliteDict('./db.sqlite',tablename=f'posiedzenie{i}', autocommit=True)
    for j in range(len(posiedzenie)):
        entry = db_posiedzenie.get(str(j), None)
        if not entry:
            print(f"Posiedzenie {i}, przemowienie {j}")
            kwords = extract_keywords(posiedzenie[j]['tekst'], llm_chain)
            print(f"Response: {kwords}")
            to_posiedzenie = posiedzenie[j]
            to_posiedzenie['keywords'] = kwords
            db_posiedzenie[str(j)] = to_posiedzenie
    db_posiedzenie.close()



# responses = {}
# for i in range(len(posiedzenie1)):
#     wypowiedz = posiedzenie1[i]['tekst']
#     texts = text_splitter.create_documents([wypowiedz])
#     responses[i] = []
#     for doc in texts:
#         response = llm_chain.run(doc.page_content)
#         print("DOKUMENT:")
#         print(doc)
#         print("SŁOWA KLUCZOWE:")
#         print(response)
#         responses[i].append(response)
#     db[str(i)] = str(responses[i])

db.close()
