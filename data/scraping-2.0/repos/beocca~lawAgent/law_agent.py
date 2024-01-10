import os
import time
import json
import random

import requests
from bs4 import BeautifulSoup

from openai.error import InvalidRequestError

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
	SystemMessage,
    HumanMessage,
    AIMessage
)

from config import *
from utils import formatting



class LawAgent:

    rechtsfrage: str
    summary: dict
    gesetze_durchsucht: list

    bundesrecht_index: dict
    prompts: dict
    messages: list
    conversation_history: list

    chat: ChatOpenAI
    chat_16k: ChatOpenAI


    def __init__(self) -> None:
        # Load Bundesrecht Index Filled
        with open(os.path.join("ris", "bundesrecht_index_filled.json"), "r") as f:
            self.bundesrecht_index = json.load(f)

        # Load Prompts
        self.prompts = {
            "system":                       self.init_prompt("01_get_gesetze", "01_system.txt"),
            "kategorie_waehlen":            self.init_prompt("01_get_gesetze", "02_kategorie_waehlen.txt"),
            "gesetz_waehlen":               self.init_prompt("01_get_gesetze", "03_gesetz_waehlen.txt"),
            "zusammenfassung_erstellen":    self.init_prompt("01_get_gesetze", "04_zusammenfassung_erstellen.txt"),
            "gesetzestext_teil_waehlen":    self.init_prompt("01_get_gesetze", "05_gesetzestext_teil_waehlen.txt"),
            "gesetzestext_teil_zeigen":     self.init_prompt("01_get_gesetze", "06_gesetzestext_teil_zeigen.txt"),
            "gesetzestext_gesamt":          self.init_prompt("01_get_gesetze", "07_gesetzestext_gesamt.txt"),
            "finaler_report":               self.init_prompt("01_get_gesetze", "08_finalen_report_erstellen.txt"),

            "extrahiere_fachbegriffe":      self.init_prompt("02_erklaere_final_report", "01_analysiere_finalen_report.txt"),
            "fragen_generieren":            self.init_prompt("02_erklaere_final_report", "02_generiere_frage_fuer_fachbegriff.txt"),
        }

        self.chat = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            max_tokens=2048
        )
        self.chat_16k = ChatOpenAI(
            model="gpt-3.5-turbo-16k",
            temperature=0,
            max_tokens=4096
        )

        self.llm_curie = OpenAI(
            model="text-curie-001",
            temperature=0,
            max_tokens=1024
        )

        self.summary = dict()
        self.gesetze_durchsucht = list()
        self.messages = list()
        self.conversation_history = list()


    def init_prompt(self, dir, prompt_name):
        with open(os.path.join(CHAIN_DIR, dir, prompt_name), "r") as f: prompt = f.read()
        return prompt
    

    def run(self, question, max_interations=5):
        # main function to answer a question

        ## INIT MAIN VARIABLES FOR AGENT
        if isinstance(question, str):
            self.rechtsfrage = question
            self.add_message(
                SystemMessage(
                    content=self.prompts["system"]\
                        .replace("{rechtsfrage}", self.rechtsfrage)\
                        .replace("{gesetze_durchsucht}", str(self.gesetze_durchsucht)) \
                        .replace("{summary}", str(self.summary))
                )
            )
            # TODO: Question interpretation and refinement -> alignment between agent and user on interpretation of question


        
        elif isinstance(question, dict):
            # TODO: init variables from previous conversation state
            raise NotImplementedError
        
        try:
            analysis = None
            final_report = None
            fachbegriffe = None
            explained_fachbegriffe = dict()
            for i in range(max_interations):

                ## CHOOSE GESETZ
                # define layers, choose gesetz, erstelle zusammenfassung und reset messages
                layers = self.define_layers()
                gesetz = self.choose_gesetz(layers)
                self.summarize_progress()
                self.reset_messages(out=True)
                if "nichts gefunden" in gesetz.lower(): continue

                ## SCRAPE GESETZ
                gesetz_id = gesetz.split(" - ")[0]
                gesetz_structure = self.get_gesetz_structure(gesetz_id)
                gesetz_is_long = len(str(gesetz_structure)) > 8096

                if gesetz_is_long:
                    ## CHOOSE SEKTION VON GESETZ            
                    # choose sektion and analyse
                    geltende_fassung = None
                    chosen_section = self.choose_section_from_gesetz(gesetz, gesetz_structure)
                    while geltende_fassung is None:

                        while chosen_section not in gesetz_structure.keys():
                            self.retry_completion()
                            print("retrying completion")

                        if isinstance(gesetz_structure[chosen_section], list):
                            t = gesetz_structure[chosen_section]
                            while len(t) == 1: t = t[0]
                            geltende_fassung = "\n".join(t)
                            
                        elif isinstance(gesetz_structure[chosen_section], dict):
                            gesetz_structure = gesetz_structure[chosen_section]
                            chosen_section = self.choose_section_from_gesetz(gesetz, gesetz_structure)
                        
                        # TODO: prevent infinite loop

                    analysis = self.analyze_section_from_gesetz(gesetz, geltende_fassung)
                    # add gesetz to gesetze_durchsucht
                    analyzed_section = analysis["analysierte_sektion"]
                    self.gesetze_durchsucht.append(f"{gesetz} - {analyzed_section}")
                
                else:
                    ## ANALYZE FULL GESETZ
                    # analyse
                    analysis = self.analyze_full_gesetz(gesetz, gesetz_structure)
                    # add gesetz to gesetze_durchsucht
                    self.gesetze_durchsucht.append(gesetz)


                ## DECISION
                naechster_schritt = analysis["naechster_schritt"]
                if naechster_schritt.lower() != "done":
                    # create summary and start over
                    self.summarize_progress()
                    self.reset_messages()
                    continue
                else:
                    # create summary and final report
                    self.summarize_progress()
                    final_report = self.create_final_report()

                    # reset messages
                    self.reset_messages()

                    ## ERKLAERE FACHBEGRIFFE
                    # extract fachbegriffe
                    fachbegriffe = self.extract_fachbegriffe(final_report)

                    # explain fachbegriffe
                    if len(fachbegriffe) > 0:
                        fragen_for_fachbegriffe = self.generate_questions_for_fachbegriffe()
                        assert isinstance(fragen_for_fachbegriffe, dict)

                        for fachbegriff in fragen_for_fachbegriffe.keys():
                            
                            # Create new law agent to answer question
                            la = LawAgent()
                            answer = la.run(fragen_for_fachbegriffe[fachbegriff]["frage"])

                            # Add answer to explained_fachbegriffe
                            explained_fachbegriffe[fachbegriff] = {
                                "frage": fragen_for_fachbegriffe[fachbegriff]["frage"],
                                "antwort": answer
                            }

                            # TODO: update final report with answer
                            # TODO: update gesetze_durchsucht with answer 

                    break



        except KeyboardInterrupt:
            # TODO: stop and summarize conversation
            # TODO: save whole conversation status
            pass
        


        # TODO: handle if final_report is None

        # save whole conversation
        save_dict = {
            "rechtsfrage": self.rechtsfrage,
            "gesetze_durchsucht": self.gesetze_durchsucht,
            "summary": self.summary,
            "last_analysis": analysis,
            "final_report": final_report,
            "fachbegriffe": explained_fachbegriffe if len(explained_fachbegriffe.keys()) == 0 else fachbegriffe,
            "conversation_history": [f"{m.type}: {m.content}" for m in self.conversation_history]
        }
        # save conversation history
        frage_str = self.rechtsfrage.replace(" ", "_").replace("?", "").replace("!", "").replace(".", "").strip()
        with open(os.path.join("answered", f"conversation_history_{frage_str}.json"), "w") as f:
            json.dump(save_dict, f, indent=4, ensure_ascii=False)

        # reset all variables after run
        self.rechtsfrage = None
        self.messages = list()
        self.conversation_history = list()
        self.gesetze_durchsucht = list()
        self.summary = dict()

        return save_dict


    def extract_fachbegriffe(self, finaler_report):

        # only keep the einfache_antwort
        assert "einfache_antwort" in finaler_report.keys()
        finaler_report = {"antwort": finaler_report["einfache_antwort"]}

        output_format = {
            "extrahierte_fachbegriffe": ["fachbegriff_1", "..."]
        }
        none_found_format = {
            "extrahierte_fachbegriffe": []
        }
        current_human_message = HumanMessage(
            content=self.prompts["extrahiere_fachbegriffe"].format(
                rechtsfrage=self.rechtsfrage,
                finaler_report=formatting.dict_to_string(finaler_report),
                output_format=
                    f"{formatting.dict_to_string(output_format)}\n"
                     "XOR wenn keine Fachbegriffe in der Antwort enhalten sind:\n"
                    f"{formatting.dict_to_string(none_found_format)}"
            )
        )

        fachbegriffe = self.get_chat_completion(current_human_message)
        assert "extrahierte_fachbegriffe" in fachbegriffe.keys()
        assert isinstance(fachbegriffe["extrahierte_fachbegriffe"], list)
        return fachbegriffe["extrahierte_fachbegriffe"]


    def generate_questions_for_fachbegriffe(self):
        
        output_format = {
            "fragen": [
                {
                    "fachbegriff": "fachbegriff_1",
                    "frage": "eine rechtsfrage die du dir selbst stellen würdest um diesen fachbegriff zu erklaeren"
                }
            ]
        }
        
        current_human_message = HumanMessage(
            content=self.prompts["fragen_generieren"].format(
                output_format=formatting.dict_to_string(output_format)
            )
        )

        questions = self.get_chat_completion(current_human_message)
        assert "fragen" in questions.keys()
        assert isinstance(questions["fragen"], list)
        return {
            e["fachbegriff"]: {
                "frage": e["frage"]
            }
            for e in questions["fragen"]
        }


    def lookup_bundesrecht(self, layers):
        if len(layers) == 0:
            return self.bundesrecht_index.keys()
        elif len(layers) == 1:
            l1 = layers[0]
            return [ k for k in self.bundesrecht_index[l1].keys() if not k.endswith(" FREI") ]
        elif len(layers) == 2:
            l1, l2 = layers
            try: return [ k for k in self.bundesrecht_index[l1][l2].keys() if not k.endswith(" FREI") ]
            except KeyboardInterrupt: raise KeyboardInterrupt
            except: return None
        else:
            raise ValueError("Too many layers.")


    def bundesrecht_gesetze_for_category(self, layers):
        assert len(layers) <= 3
        
        first, second, third = layers

        if third is None:
            gesetze = self.bundesrecht_index[first][second]
            assert isinstance(gesetze, list)
            return gesetze 
        else:
            gesetze = self.bundesrecht_index[first][second][third]
            assert isinstance(gesetze, list)
            return gesetze
        


    def define_layers(self):
        
        # Define initial variables
        layers = list()

        ## Set layers
        layers = self.define_layer(layers, f"Zu beantwortende Rechtsfrage: {self.rechtsfrage}")  # first layer
        assert len(layers) == 1
        assert layers[0] in self.bundesrecht_index.keys()
        layers = self.define_layer(layers, f"Du hast {layers[0]} gewaehlt.")                      # second layer
        assert len(layers) == 2
        assert layers[1] in self.bundesrecht_index[layers[0]].keys()
        layers = self.define_layer(layers, f"Du hast {layers[1]} gewaehlt.")                      # third layer
        assert 2 <= len(layers) <= 3

        if layers[-1] is not None:
            assert isinstance(self.bundesrecht_index[layers[0]][layers[1]], dict)
            assert isinstance(self.bundesrecht_index[layers[0]][layers[1]][layers[2]], list)

        return layers


    def define_layer(self, layers, context):

        next_layer = self.lookup_bundesrecht(layers)
        if next_layer is None: return layers + [None]
        elif len(next_layer) == 0: return layers + [None]
        assert len(next_layer) > 0

        # get 2 random choices if possible
        if len(next_layer) < 2: rand = next_layer[1]
        else:
            random_choices = random.sample(next_layer, 2)
            rand = f"{random_choices[0]}, {random_choices[1]}, ..."
        
        output_format = {"kategorie": f"gewaehlte Kategorie inklusive voranstehende Zahl. z.B. {rand}"}
        
        # Define human message
        current_human_message = HumanMessage(
            content=self.prompts["kategorie_waehlen"].format(
                context=context,
                categories="\n".join(next_layer),
                output_format=formatting.dict_to_string(output_format)
            )
        )
        response = self.get_chat_completion(current_human_message)

        # Define chosen layers and return
        chosen_category = response["kategorie"]
        if len(layers) == 0: layers = [chosen_category]
        else: layers.append(chosen_category)
        return layers


    def summarize_progress(self):
        output_format = {
            "zusammenfassung": "eine kurze, aber detailierte zusammenfassung ueber deinen bisherigen Fortschritt",
            "frage_beantwortet": "hast du die frage schon beantwortet? waehle aus folgender liste: 'ja' | 'noch nicht' ",
            "begruendung": "begruende deine Entscheidung",
        }
        current_human_message = HumanMessage(
            content=self.prompts["zusammenfassung_erstellen"].format(
                output_format=formatting.dict_to_string(output_format)
            )
        )

        summary = self.get_chat_completion(current_human_message)
        self.summary = summary

    
    def choose_gesetz(self, layers):
        # Choose gesetz to look through

        # zusammenfassung = self.summary["zusammenfassung"]
        context = f"Zu beantwortende Rechtsfrage: {self.rechtsfrage}"  #\n\nZusammenfassung des bisherigen Fortschritts: {zusammenfassung}"

        gesetze = [
            g["gesetzesnummer"] + " - " + g["kurztitel"].replace(" - ", "; ")
            for g in self.bundesrecht_gesetze_for_category(layers)
            if len(str(g["gesetzesnummer"]).strip()) > 0
        ]
        output_format = {"nummer": "die davorstehende nummer des gesetzes oder 'nichts gefunden'", "titel": "der titel des gewählten gesetzes oder 'nichts gefunden'"}
        current_human_message = HumanMessage(
            content=self.prompts["gesetz_waehlen"].format(
                context=context,
                laws="\n".join(gesetze),
                gesetze_durchsucht="\n".join(self.gesetze_durchsucht),
                output_format=formatting.dict_to_string(output_format)
            )
        )
        response = self.get_chat_completion(current_human_message)
        if "nichts gefunden" in response["nummer"].lower() or "nichts gefunden" in response["titel"].lower():
            return "nichts gefunden"
        else:
            gesetz = response["nummer"] + " - " + response["titel"]
            return gesetz




    def choose_section_from_gesetz(self, gesetz, gesetz_structure):
        context = f"Zu beantwortende Rechtsfrage: {self.rechtsfrage}\n\nZusammenfassung des bisherigen Fortschritts: {self.summary['zusammenfassung']}"
        output_format = {
            "gewaehlte_sektionen": ["sektion (ganze zeile zitiert!!)", "..." ]
        }
        choose_section_message = HumanMessage(
            content=self.prompts["gesetzestext_teil_waehlen"].format(
                context=context,
                gesetz=gesetz,
                struktur="\n".join([s for s in gesetz_structure.keys()]),
                output_format=formatting.dict_to_string(output_format)
            ) + "\n\nAchte darauf, dass du immer die gesamte Zeile zitierst und nicht nur die Nummer der Sektion!"
        )
        response = self.get_chat_completion(choose_section_message, model="16k")
        chosen_sections = response["gewaehlte_sektionen"]

        # chosen_sections = [s["paragraph"] + " - " + s["name"] for s in chosen_sections]

        return chosen_sections[0]  # TODO: return all chosen sections
    



    def analyze_full_gesetz(self, gesetz, gesetz_structure):
        geltende_fassung = str()
        for k, v in gesetz_structure.items():
            content = " ".join(v)
            geltende_fassung += f"{k}\n"
            geltende_fassung += f"{content}\n\n"
        geltende_fassung = geltende_fassung.strip()

        output_format = {
            "vermutung": "stelle eine Vermutungen an ob dieses Gesetz ausreichend ist um die Frage zu beantworten? waehle aus folgender liste: 'ja' | 'nein'",
            "begruendung": "eine kurze begruendung warum",
            "loesungsansatz": "wie koennte die frage beantwortet werden?",
            "naechster_schritt": "was sollte als naechstes getan werden? waehle aus folgender liste: 'neues gesetz waehlen' | 'done' "
        }
        show_chosen_section_message = HumanMessage(
            content=self.prompts["gesetzestext_gesamt"].format(
                gesetz=gesetz,
                geltende_fassung=geltende_fassung,
                output_format=formatting.dict_to_string(output_format)
            )
        )

        # get chat completion and return analysis of gesetz
        analysis = self.get_chat_completion(show_chosen_section_message, model="16k")
        return analysis

    


    def analyze_section_from_gesetz(self, gesetz, geltende_fassung):

        output_format = {
            "vermutung": "stelle eine Vermutungen an ob der gebene Teil ausreichend ist um die Frage zu beantworten? waehle aus folgender liste: 'ja' | 'nein'",
            "begruendung": "eine kurze begruendung warum",
            "analysierte_sektion": "der name oder id der sektion die analysiert wurde",
            "loesungsansatz": "wie koennte die frage beantwortet werden?",
            "naechster_schritt": "was sollte als naechstes getan werden? waehle aus folgender liste: 'neues gesetz waehlen' | 'done' "
        }
        show_chosen_section_message = HumanMessage(
            content=self.prompts["gesetzestext_teil_zeigen"].format(
                gesetz=gesetz,
                geltende_fassung=geltende_fassung,
                output_format=formatting.dict_to_string(output_format)
            )
        )

        # get chat completion and return analysis of gesetz
        analysis = self.get_chat_completion(show_chosen_section_message, model="16k")
        return analysis
        

    
    def create_final_report(self):
        output_format = {
            "zusammenfassung": "fasse noch einmal zusammen wie du beim beantworten der Frage vorgegangen bist",
            "komplexe_antwort": "gib eine möglichst genaue und komplexe antwort und erklaerung; zusätzliche informationen sind gerne gesehen; gerichtet an einen juristischen Experten",
            "einfache_antwort": "gib eine einfache und kurze antwort; vermeide informationen nach welchen nicht explizit gefragt wird, sowie Fachjargon; gerichtet an einen juristischen Laien",
            "begruendung": "begruende deine antwort",
        }

        current_human_message = HumanMessage(
            content=self.prompts["finaler_report"].format(
                output_format=formatting.dict_to_string(output_format)
            )
        )

        # get chat completion and return the final report
        final_report = self.get_chat_completion(current_human_message, model="16k")
        return final_report
    

    def retry_completion(self):
        
        # specify human message so the law agent tries again
        current_human_message = HumanMessage(
            content=\
                "Thanks a lot! But your output does not follow the specified format. "\
                "Please try again. Do not explain yourself and do not give excuses. "\
                "Make sure that your answer matches the previously specified output format exactly!"
        )

        # get chat completion and return the response
        response = self.get_chat_completion(current_human_message, model="16k")
        return response


    
    def get_chat_completion(self, human_message, model="4k"):
        assert model in ["4k", "16k"]

        # clean human message
        human_message = HumanMessage(
            content=formatting.clean_text_for_prompt(human_message.content)  + f"\n\nDeine JSON-Antwort:"
        )

        # append human message to conversation history and agent memory
        self.add_message(human_message)
        
        # get chat completion
        try:
            if model == "4k": response = self.chat(self.messages)
            if model == "16k": response = self.chat_16k(self.messages)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except InvalidRequestError:
            response = self.chat_16k(self.messages)

        # append response message to conversation history and agent memory
        self.add_message(response)

        # do checks and return 
        time.sleep(1)
        assert isinstance(response, AIMessage)
        return json.loads(response.content)
        # TODO: handle this better! -> i.e. return a message that the agent did not understand the human message


    def reset_messages(self, out=False):

        if out: 
            for m in self.messages:
                assert isinstance(m, SystemMessage) or isinstance(m, HumanMessage) or isinstance(m, AIMessage)
                if isinstance(m, SystemMessage):    u = "System"
                if isinstance(m, HumanMessage):     u = "Human"
                if isinstance(m, AIMessage):        u = "AI"
                print(u, m.content)

        self.messages = self.messages[:1]


    def add_message(self, message):
        self.messages.append(message)
        self.conversation_history.append(message)
    

    def get_gesetz_structure(self, gesetz_id):

        # Get Geltende Fassung von Gesetz
        gesetz_structure_path = os.path.join("ris", "bundesrecht", f"gesetz_structure_{gesetz_id}.json")
        if not os.path.exists(gesetz_structure_path):
            url = f"https://www.ris.bka.gv.at/GeltendeFassung.wxe?Abfrage=Bundesnormen&Gesetzesnummer={gesetz_id}"
            html = requests.get(url).text
            soup = BeautifulSoup(html, "html.parser")
            
            pagebase = soup.find("div", {"id": "pagebase"})
            content = pagebase.find("div", {"id": "content"})
            document_contents = content.find_all("div", {"class": "documentContent"})
            
            gesetz_structure = {}
            curr_ueberschr_g1 = None
            curr_ueberschr_para = None
            curr_gld_symbol = None
            for doc_content in document_contents:
                
                # old version
                # text_nodes = doc_content.find_all(string=True)
                # text_nodes = [tn.strip() for tn in text_nodes if len(tn.strip()) > 0]
                # text_nodes = [tn for tn in text_nodes if tn.lower() != "text"]
                # text_nodes = [tn for tn in text_nodes if not tn.startswith("Art. ")]
                # text_nodes = [formatting.clean_text_for_prompt(tn) for tn in text_nodes]
                # section_name, section_content = " - ".join(text_nodes[:2]), text_nodes[2:]
                # section_content = [c for c in section_content if not c.startswith("§")]
                # section_content = [c for c in section_content if not c.startswith("Paragraph ")]
                # section_content = [formatting.clean_text_for_prompt(c) for c in section_content]
                # gesetz_structure[section_name.replace(" ", "")] = text_nodes[2:]

                # new version
                # gesetz_structure = {
                #   "ueberschr_g1": {
                #       "ueberschr_para": {
                #           "gld_symbol": [ (title), absatz_text, absatz_text, ... ]
                #        },
                #       "gld_symbol": [ (title), absatz_text, absatz_text, ... ]
                #   }
                # }

                # <h4 class="UeberschrG1 AlignCenter">
                ueberschr_g1 = doc_content.find_all("h4", {"class": "UeberschrG1"})
                if len(ueberschr_g1) > 0:
                    assert len(ueberschr_g1) == 1
                    curr_ueberschr_g1 = ueberschr_g1[0].text.strip()
                    if curr_ueberschr_g1 not in gesetz_structure.keys(): 
                        gesetz_structure[curr_ueberschr_g1] = {}
                    
                    curr_ueberschr_para = None
                    curr_gld_symbol = None


                # <h4 class="UeberschrPara AlignCenter">Abstammung</h4>
                ueberschr_para = doc_content.find_all("h4", {"class": "UeberschrPara"})
                if len(ueberschr_para) > 0:
                    assert len(ueberschr_para) == 1
                    curr_ueberschr_para = formatting.key_formatting_for_dict(ueberschr_para[0].text.strip())
                    if curr_ueberschr_para not in gesetz_structure[str(curr_ueberschr_g1)].keys():
                        gesetz_structure[curr_ueberschr_g1][curr_ueberschr_para] = {}

                    curr_gld_symbol = None

                # # <div class="ParagraphMitAbsatzzahl">
                # para_mit_abs = doc_content.find_all("div", {"class": "ParagraphMitAbsatzzahl"})
                # if len(para_mit_abs) > 0:
                #     assert len(para_mit_abs) == 1
                #     curr_para_mit_abs = para_mit_abs[0]

                    
                # <div class="MarginTop4 AlignJustify">
                gld_symbol = doc_content.find_all("div", {"class": "MarginTop4"})
                if len(gld_symbol) > 0:
                    # <span class="sr-only">Paragraph 8,</span>
                    text = gld_symbol[0].find_all("span", {"class": "sr-only"})
                    assert len(text) > 0
                    curr_gld_symbol = formatting.key_formatting_for_dict(text[0].text)

                else:
                    # <h5 class="GldSymbol AlignJustify">
                    gld_symbol = doc_content.find_all("h5", {"class": "GldSymbol"})
                    if len(gld_symbol) > 0:
                        curr_gld_symbol = formatting.key_formatting_for_dict(gld_symbol[0].find_all("span", {"class": "sr-only"})[0].text)
                    
                    

                wai_absatz_list = doc_content.find_all("ol", {"class": "wai-absatz-list"})
                wai_list = doc_content.find_all("ol", {"class": "wai-list"})
                top = doc_content.find_all("div", {"class": "MarginTop4"})
                if len(wai_absatz_list) > 0:
                    law_text = []
                    for wal in wai_absatz_list:
                        lis = wai_absatz_list[0].find_all("li")
                        
                        for li in lis:
                            absatz_zahl = li.find_all("span", {"class": "sr-only"})[0].text
                            law_text.append(li.text)
                
                elif len(wai_list) > 0:
                    law_text = []
                    for wl in wai_list:
                        lis = wl.find_all("li")
                        # try to find title for list
                        # <div class="MarginTop4 AlignJustify">
                        if len(top) > 0:
                            law_text.append([top[0].text] + [ li.text for li in lis])
                        else:
                            law_text.append([ li.text for li in lis])

                # elif len(top) > 0:
                #     law_text = top[0].text.strip()
                #     law_text = [law_text]
                else:
                    law_text = [doc_content.text.strip()]

                # TODO: add to gesetz_structure
                # ...
                if curr_ueberschr_g1 is not None:
                    if curr_ueberschr_para is not None:
                        if isinstance(gesetz_structure[curr_ueberschr_g1][curr_ueberschr_para], dict):
                            if curr_gld_symbol not in gesetz_structure[curr_ueberschr_g1][curr_ueberschr_para].keys():
                                gesetz_structure[curr_ueberschr_g1][curr_ueberschr_para][curr_gld_symbol] = [law_text]
                            else:
                                gesetz_structure[curr_ueberschr_g1][curr_ueberschr_para][curr_gld_symbol].append(law_text)
                        else:
                            if curr_gld_symbol not in gesetz_structure[curr_ueberschr_g1].keys():
                                gesetz_structure[curr_ueberschr_g1][curr_gld_symbol] = [law_text]
                            else:
                                gesetz_structure[curr_ueberschr_g1][curr_gld_symbol].append(law_text)
                    else:
                        if curr_gld_symbol not in gesetz_structure[curr_ueberschr_g1].keys():
                            gesetz_structure[curr_ueberschr_g1][curr_gld_symbol] = [law_text]
                        else:
                            gesetz_structure[curr_ueberschr_g1][curr_gld_symbol].append(law_text)
                else:
                    continue


            # # create cleaner gesetz structure
            # gesetz_structure = self.structure_gesetz_helper(gesetz_structure)

            # save as json
            with open(gesetz_structure_path, "w") as f:
                json.dump(gesetz_structure, f, indent=4, ensure_ascii=False)

        else:
            with open(gesetz_structure_path, "r") as f:
                gesetz_structure = json.load(f, strict=False)

        return gesetz_structure
    


    def structure_gesetz_helper(self, gesetz_structure):

        # load prompt
        prompt = None
        with open(os.path.join("chains", "00_helper", "clean_gesetz_section_title.txt"), "r") as f:
            prompt = f.read()
        assert prompt is not None

        # initialize variables for loop
        cleaned_section_names = []
        section_names = list(gesetz_structure.keys())

        # loop over all section names
        for i, section_name in enumerate(section_names):

            # format prompt and get response
            formatted_prompt = prompt.format(eingabe=section_name)
            response = self.llm_curie.generate([formatted_prompt]).generations[0][0].text
            
            # clean response
            response = response.split("\n")[0]
            response = response.strip()

            # append cleaned response to list
            cleaned_section_names.append(response)

        # Create new gesetz structure and return it
        new_gesetz_structure = {}
        assert len(cleaned_section_names) == len(gesetz_structure.keys())
        for old, new in zip(gesetz_structure.keys(), cleaned_section_names):
            new_gesetz_structure[new] = gesetz_structure[old.replace(" ", "")]
            

        return new_gesetz_structure
    


    def format_gesetz_structure(self):
        pass


    def format_gesetz_section_content(self):
        pass








if __name__ == "__main__":
    
    la = LawAgent()
    fragen = [
        "Welche Voraussetzungen müssen erfüllt sein, damit eine Person in Österreich die Staatsbürgerschaft erlangen kann?",
        # "Welche Behörde ist in Österreich für die Registrierung von Unternehmen zuständig und welche Schritte sind erforderlich, um ein Unternehmen rechtlich anzumelden?",
        # "Wie schnell darf ich auf der Autobahn mit einem Fahrrad fahren?",
        # "Was sind die rechtlichen Bestimmungen für die Kündigung eines Arbeitsvertrags in Österreich und welche Rechte haben Arbeitnehmer und Arbeitgeber in diesem Zusammenhang?",
        # "Welche gesetzlichen Regelungen gelten in Österreich für den Schutz des geistigen Eigentums, insbesondere für Markenrechte und Urheberrechte?",
        # "Wie lange darf ein sich ein 15 jähriger in der Nacht draußen aufhalten?",
        # "Welche steuerrechtlichen Regelungen gelten in Österreich für die Besteuerung von Einkommen aus dem Verkauf von Immobilien und wie hoch ist der Steuersatz?"
    ]
    for frage in fragen:
        la.run(frage)


    print("...done")



