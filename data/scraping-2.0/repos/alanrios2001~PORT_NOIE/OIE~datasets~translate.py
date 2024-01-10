from src.conll2bioes import Conversor
import os
import spacy
from tqdm.auto import tqdm
from main import criar_conll
import typer
from deep_translator import GoogleTranslator
import json
import pathlib
from diskcache import Cache
from OIE.datasets.validated_splits.contractions import transform_portuguese_contractions, clean_extraction
from OIE.final.matcher import OIE_Match
import openai
import httpx
import time


app = typer.Typer()


class LoadDataset:
    def __init__(self,
                 dataset_path: str,
                 dataset_name: str,
                 out_path: str
                 ):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path

        with open(self.dataset_path +"/"+ self.dataset_name, "r", encoding="utf-8") as f:
            data = f.read()

        # selecionando apenas exts com arg0 rel e arg1
        data = data.split("\n\t")

        data_norm = []
        for ext in data:
            if "ARG5" not in ext:
                if "ARG4" not in ext:
                    if "ARG3" not in ext:
                        if "ARG2" not in ext:
                            if "ARG1" in ext:
                                if "V" in ext:
                                    if "ARG0" in ext:
                                        data_norm.append(ext)


        path = out_path + "/mod"
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        with open(path + "/" + dataset_name, "a", encoding="utf-8") as f:
            raw = data_norm
            raw = "\n\t".join(raw)
            f.write(raw)

        Conversor(path+"/", dataset_name, out_path)


class ArgsRel2:
    def __init__(self):
        self.provavel_rel = []
        self.alinhamentos = []
        try:
            self.nlp = spacy.load("pt_core_news_lg")
        except:
            os.system("python -m spacy download pt_core_news_lg")
            self.nlp = spacy.load("pt_core_news_lg")

    def root_parse(self, doc_dict, root_idx):
        #encontra centro da extração pelo root
        for idx in doc_dict:
            pos = doc_dict[idx]["pos"]
            dep = doc_dict[idx]["dep"]
            if (pos == "VERB" and dep == "ROOT") and (idx != 0 and idx != len(doc_dict) - 1):#restringe primeiro e último caracter da frase inteira
                self.provavel_rel.append("VERB")
                return (idx, idx)
        root_idx = self.verb_parse(doc_dict, root_idx)
        return root_idx

    def verb_parse(self, doc_dict, root_idx):
        #encontra centro da extração pelo root
        for idx in doc_dict:
            pos = doc_dict[idx]["pos"]
            dep = doc_dict[idx]["dep"]
            if (pos == "VERB" and (dep == "xcomp" or dep == "acl" or dep == "acl:relacl")) and (idx != 0 and idx != len(doc_dict) - 1):#restringe primeiro e último caracter da frase inteira
                self.provavel_rel.append("VERB")
                return (idx, idx)

        root_idx = self.aux_parse(doc_dict, root_idx)
        return root_idx



    def aux_parse(self, doc_dict, root_idx):
        #encontra centro da extração pelo root
        for idx in doc_dict:
            pos = doc_dict[idx]["pos"]
            dep = doc_dict[idx]["dep"]
            if (pos == "AUX" and dep == "ROOT") and (idx != 0 and idx != len(doc_dict) - 1):#restringe primeiro e último caracter da frase inteira
                self.provavel_rel.append("AUX")
                return (idx, idx)

        root_idx = self.aux_parse2(doc_dict, root_idx)
        return root_idx


    def aux_parse2(self, doc_dict, root_idx):
        #encontra centro da extração pelo root
        for idx in doc_dict:
            pos = doc_dict[idx]["pos"]
            dep = doc_dict[idx]["dep"]
            if (pos == "AUX" and dep == "cop") and (idx != 0 and idx != len(doc_dict) - 1):#restringe primeiro e último caracter da frase inteira
                self.provavel_rel.append("AUX")
                return (idx, idx)
        root_idx = self.noun_parse(doc_dict, root_idx)
        return root_idx

    def noun_parse(self, doc_dict, root_idx):
        #encontra centro da extração pelo root
        for idx in doc_dict:
            pos = doc_dict[idx]["pos"]
            dep = doc_dict[idx]["dep"]
            if (pos == "NOUN" and dep == "ROOT") and (idx != 0 and idx != len(doc_dict) - 1):#restringe primeiro e último caracter da frase inteira
                self.provavel_rel.append("NOUN")
                return (idx, idx)
        return root_idx


    def get_args_rel(self, ext, sent):
        self.alinhamentos = []
        doc = self.nlp(ext)
        doc_dict = {}
        i = 0
        for token in doc:
            doc_dict[i] = {"text": token.text, "pos": token.pos_, "dep": token.dep_}
            i += 1
        root_idx = (None, None)
        self.provavel_rel = []
        root_idx = self.root_parse(doc_dict, root_idx)

        if len(self.provavel_rel)>0 and self.provavel_rel[0] == "VERB":
            if root_idx[0]-1 != 0:
                if doc_dict[root_idx[0]-1]["pos"] in ["AUX",'ADV']:
                    root_idx = (root_idx[0]-1, root_idx[1])

        #verificando elementos que compoem a rel depois do centro
        if root_idx != (None, None):
            for j in range(root_idx[1]+1, len(doc_dict)):
                pos = doc_dict[j]["pos"]
                self.provavel_rel.append(pos)

        adp_idxs = []
        for idx, pos_ in enumerate(self.provavel_rel[1:-1]):
            if pos_ in ['ADJ','ADV','NOUN', 'VERB','ADV']:
                continue
            elif pos_ == 'ADP':
                adp_idxs.append(idx+1)
                continue
            else:
                break
        adp_idxs.append(0)

        for idx in adp_idxs:
            arg1 = ""
            rel = ""
            arg2 = ""
            if root_idx != (None, None):
                new_root_idx = (root_idx[0],root_idx[1]+idx)
                j = new_root_idx[0]
                while j <= new_root_idx[1]:
                    rel += doc_dict[j]["text"] + " "
                    j += 1

                for idx in doc_dict:
                    token = doc_dict[idx]["text"]
                    if idx < new_root_idx[0]:
                        arg1 += token + " "
                    if idx > new_root_idx[1]:
                        arg2 += token + " "

            self.alinhamentos.append((arg1,rel,arg2))


        return self.alinhamentos

class ArgsRel:
    def __init__(self):
        self.current_root_sint = None
        self.alinhamentos = []
        try:
            self.nlp = spacy.load("pt_core_news_lg")
        except:
            os.system("python -m spacy download pt_core_news_lg")
            self.nlp = spacy.load("pt_core_news_lg")

    def root_parse(self, doc_dict, root_idx):
        #encontra centro da extração pelo root
        for idx in doc_dict:
            pos = doc_dict[idx]["pos"]
            dep = doc_dict[idx]["dep"]
            if (pos == "VERB" and dep == "ROOT") and (idx != 0 and idx != len(doc_dict) - 1):
                root_idx = (idx, idx)
                self.current_root_sint = "VERB-ROOT"
                break
        if root_idx == (None, None):
            root_idx = self.aux_parse(doc_dict, root_idx)
        return root_idx

    def aux_parse(self, doc_dict, root_idx):
        #encontra centro da extração pelo root
        for idx in doc_dict:
            pos = doc_dict[idx]["pos"]
            dep = doc_dict[idx]["dep"]
            if (pos == "AUX" and dep == "cop") and (idx != 0 and idx != len(doc_dict) - 1):
                root_idx = (idx, idx)
                self.current_root_sint = "AUX-cop"
                break
            elif (pos == "AUX" and dep == "ROOT") and (idx != 0 and idx != len(doc_dict) - 1):
                root_idx = (idx, idx)
                self.current_root_sint = "AUX-ROOT"
                break
        if root_idx == (None, None):
            root_idx = self.x_comp_parse(doc_dict, root_idx)
        return root_idx
    def x_comp_parse(self, doc_dict, root_idx):
        #encontra centro da extração pelo root
        for idx in doc_dict:
            pos = doc_dict[idx]["pos"]
            dep = doc_dict[idx]["dep"]
            if (pos == "VERB" and dep == "xcomp" and (idx != 0 and idx != len(doc_dict) - 1)):
                root_idx = (idx, idx)
                self.current_root_sint = "VERB-xcomp"
                break
            elif (pos == "VERB" and dep == "acl" and (idx != 0 and idx != len(doc_dict) - 1)):
                root_idx = (idx, idx)
                self.current_root_sint = "VERB-acl"
                break
            elif (pos == "VERB" and dep == "acl:relcl" and (idx != 0 and idx != len(doc_dict) - 1)):
                root_idx = (idx, idx)
                self.current_root_sint = "VERB-acl:relacl"
                break
        if root_idx == (None, None):
            root_idx = self.noun_root_parse(doc_dict, root_idx)
        return root_idx

    def noun_root_parse(self, doc_dict, root_idx):
        #encontra centro da extração pelo root
        for idx in doc_dict:
            pos = doc_dict[idx]["pos"]
            dep = doc_dict[idx]["dep"]
            if (pos == "NOUN" and dep == "ROOT" and (idx != 0 and idx != len(doc_dict) - 1)):
                root_idx = (idx, idx)
                self.current_root_sint = "NOUN-ROOT"
                break
        return root_idx

    def get_args_rel(self, ext, sent):
        self.alinhamentos = []
        doc = self.nlp(ext)
        doc_dict = {}
        i = 0
        for token in doc:
            doc_dict[i] = {"text": token.text, "pos": token.pos_, "dep": token.dep_}
            i += 1
        arg1 = ""
        rel = ""
        arg2 = ""
        root_idx = (None, None)
        self.current_root_sint = None
        root_idx = self.root_parse(doc_dict, root_idx)


        #verificando elementos que compoem a rel antes do centro
        if root_idx != (None, None):
            before_root_pos_dep = ""
            for i in range(0, root_idx[0]):
                pos = doc_dict[i]["pos"]
                dep = doc_dict[i]["dep"]
                before_root_pos_dep += pos + "-" + dep + ", "
            before_root_pos_dep = before_root_pos_dep[:-2]
            splited = before_root_pos_dep.split(", ")

            if self.current_root_sint == "NOUN-ROOT":
                if "PRON-expl" in before_root_pos_dep and splited[-1] == "PRON-expl":
                    if root_idx[0]-1 > 0:
                        root_idx = (root_idx[0]-1, root_idx[1])
                    else:
                        root_idx = (root_idx[0], root_idx[1])

            if "AUX-cop" in before_root_pos_dep and splited[-1] == "AUX-cop":
                if root_idx[0]-1 > 0:
                    root_idx = (root_idx[0]-1, root_idx[1])
                else:
                    root_idx = (root_idx[0], root_idx[1])
            elif "AUX-cop, ADV-advmod" in before_root_pos_dep and splited[-1] == "ADV-advmod":
                if root_idx[0]-2 > 0:
                    root_idx = (root_idx[0]-2, root_idx[1])
                else:
                    root_idx = (root_idx[0]-1, root_idx[1])
            elif "ADV-advmod" in before_root_pos_dep and splited[-1] == "ADV-advmod":
                if root_idx[0]-1 > 0:
                    root_idx = (root_idx[0]-1, root_idx[1])
                else:
                    root_idx = (root_idx[0], root_idx[1])
            elif "AUX-aux" in before_root_pos_dep and splited[-1] == "AUX-aux":
                if root_idx[0]-1 > 0:
                    root_idx = (root_idx[0]-1, root_idx[1])
                else:
                    root_idx = (root_idx[0], root_idx[1])
            elif "AUX-aux:pass" in before_root_pos_dep and splited[-1] == "AUX-aux:pass":
                if root_idx[0]-1 > 0:
                    root_idx = (root_idx[0]-1, root_idx[1])
                else:
                    root_idx = (root_idx[0], root_idx[1])
            elif "AUX-aux:pass" in before_root_pos_dep and splited[-1] == "AUX-aux:pass":
                if root_idx[0]-1 > 0:
                    root_idx = (root_idx[0]-1, root_idx[1])
                else:
                    root_idx = (root_idx[0], root_idx[1])
            elif "ADV-advmod, PRON-obj" in before_root_pos_dep and splited[-1] == "PRON-obj":
                if root_idx[0]-2 > 0:
                    root_idx = (root_idx[0]-2, root_idx[1])
                else:
                    root_idx = (root_idx[0]-1, root_idx[1])
            elif "AUX-cop, ADP-case" in before_root_pos_dep and splited[-1] == "ADP-case":
                if root_idx[0]-2 > 0:
                    root_idx = (root_idx[0]-2, root_idx[1])
                else:
                    root_idx = (root_idx[0]-1, root_idx[1])
            elif "AUX-cop, DET-det" in before_root_pos_dep and splited[-1] == "DET-det":
                if root_idx[0]-2 > 0:
                    root_idx = (root_idx[0]-2, root_idx[1])
                else:
                    root_idx = (root_idx[0]-1, root_idx[1])

        #verificando elementos que compoem a rel depois do centro
        if root_idx != (None, None):
            after_root_pos_dep = ""
            for i in range(root_idx[1]+1, len(doc_dict)):
                pos = doc_dict[i]["pos"]
                dep = doc_dict[i]["dep"]
                after_root_pos_dep += pos + "-" + dep + ", "
            after_root_pos_dep = after_root_pos_dep[:-2]
            splited = after_root_pos_dep.split(", ")

            if self.current_root_sint == "AUX-cop":
                if "DET-det, NOUN-ROOT, ADJ-amod, ADP-case" in after_root_pos_dep and splited[0] == "DET-det":
                    if root_idx[1]+4 < len(doc_dict) - 1:
                        root_idx = (root_idx[0], root_idx[1]+4)
                    else:
                        root_idx = (root_idx[0], root_idx[1])
            if "ADP-case, DET-det, ADV-obl, VERB-xcomp" in after_root_pos_dep and splited[0] == "ADP-case":
                if root_idx[1]+4 < len(doc_dict) - 1:
                    root_idx = (root_idx[0], root_idx[1]+4)
                else:
                    root_idx = (root_idx[0], root_idx[1]+3)
            elif "ADJ-amod, ADP-case" in after_root_pos_dep and splited[0] == "ADJ-amod":
                if root_idx[1]+2 < len(doc_dict) - 1:
                    root_idx = (root_idx[0], root_idx[1]+2)
                else:
                    root_idx = (root_idx[0], root_idx[1]+1)
            elif "VERB-xcomp, DET-det, NOUN-obj, ADP-case" in after_root_pos_dep and splited[0] == "VERB-xcomp":
                if root_idx[1]+4 < len(doc_dict) - 1:
                    root_idx = (root_idx[0], root_idx[1]+4)
                else:
                    root_idx = (root_idx[0], root_idx[1]+3)
            elif "VERB-xcomp, SCONJ-mark, VERB-xcomp, ADP-case" in after_root_pos_dep and splited[0] == "VERB-xcomp":
                if root_idx[1]+4 < len(doc_dict) - 1:
                    root_idx = (root_idx[0], root_idx[1]+4)
                else:
                    root_idx = (root_idx[0], root_idx[1]+3)
            elif "VERB-xcomp, ADP-case" in after_root_pos_dep and splited[0] == "VERB-xcomp":
                if root_idx[1]+2 < len(doc_dict) - 1:
                    root_idx = (root_idx[0], root_idx[1]+2)
                else:
                    root_idx = (root_idx[0], root_idx[1]+1)
            elif "VERB-xcomp, VERB-xcomp" in after_root_pos_dep and splited[0] == "VERB-xcomp":
                if root_idx[1]+2 < len(doc_dict) - 1:
                    root_idx = (root_idx[0], root_idx[1]+2)
                else:
                    root_idx = (root_idx[0], root_idx[1]+1)
            elif "VERB-xcomp, SCONJ-mark, VERB-xcomp" in after_root_pos_dep and splited[0] == "VERB-xcomp":
                if root_idx[1]+3 < len(doc_dict) - 1:
                    root_idx = (root_idx[0], root_idx[1]+3)
                else:
                    root_idx = (root_idx[0], root_idx[1]+2)
            elif "VERB-xcomp, VERB-xcomp, DET-det, NOUN-obj, ADP-case" in after_root_pos_dep and splited[0] == "VERB-xcomp":
                if root_idx[1]+5 < len(doc_dict) - 1:
                    root_idx = (root_idx[0], root_idx[1]+5)
                else:
                    root_idx = (root_idx[0], root_idx[1]+4)

            elif "ADJ-amod, ADP-case" in after_root_pos_dep and splited[0] == "ADJ-amod":
                if root_idx[1]+2 < len(doc_dict) - 1:
                    root_idx = (root_idx[0], root_idx[1]+2)
                else:
                    root_idx = (root_idx[0], root_idx[1]+1)
            elif "ADV-advmod, ADP-case" in after_root_pos_dep and splited[0] == "ADV-advmod":
                if root_idx[1]+2 < len(doc_dict) - 1:
                    root_idx = (root_idx[0], root_idx[1]+2)
                else:
                    root_idx = (root_idx[0], root_idx[1]+1)
            elif "ADP-case, NOUN-obj, ADP-case" in after_root_pos_dep and splited[0] == "ADP-case":
                if root_idx[1]+3 < len(doc_dict) - 1:
                    root_idx = (root_idx[0], root_idx[1]+3)
                else:
                    root_idx = (root_idx[0], root_idx[1]+2)
            elif "ADV-advmod, ADV-advmod, SCONJ-dep" in after_root_pos_dep and splited[0] == "ADV-advmod":
                if root_idx[1]+3 < len(doc_dict) - 1:
                    root_idx = (root_idx[0], root_idx[1]+3)
                else:
                    root_idx = (root_idx[0], root_idx[1]+2)
            elif "VERB-xcomp" in after_root_pos_dep and splited[0] == "VERB-xcomp":
                if root_idx[1]+1 < len(doc_dict) - 1:
                    root_idx = (root_idx[0], root_idx[1]+1)
                else:
                    root_idx = (root_idx[0], root_idx[1])
            elif "ADP-case" in after_root_pos_dep and splited[0] == "ADP-case":
                if root_idx[1]+1 < len(doc_dict) - 1:
                    root_idx = (root_idx[0], root_idx[1]+1)
                else:
                    root_idx = (root_idx[0], root_idx[1])
            elif "AUX-cop" in after_root_pos_dep and splited[0] == "AUX-cop":
                if root_idx[1]+1 < len(doc_dict) - 1:
                    root_idx = (root_idx[0], root_idx[1]+1)
                else:
                    root_idx = (root_idx[0], root_idx[1])
            elif "DET-case" in after_root_pos_dep and splited[0] == "DET-case":
                if root_idx[1]+1 < len(doc_dict) - 1:
                    root_idx = (root_idx[0], root_idx[1]+1)
                else:
                    root_idx = (root_idx[0], root_idx[1])

        j = root_idx[0]
        if root_idx != (None, None):
            while j <= root_idx[1]:
                rel += doc_dict[j]["text"] + " "
                j += 1

            for idx in doc_dict:
                token = doc_dict[idx]["text"]
                if idx < root_idx[0]:
                    arg1 += token + " "
                if idx > root_idx[1]:
                    arg2 += token + " "
        self.alinhamentos.append((arg1, rel, arg2))


        return self.alinhamentos


class ArgsRel3:
    def __init__(self):
        self.provavel_rel = []
        self.alinhamentos = []
        self.matcher = OIE_Match()
        try:
            self.nlp = spacy.load("pt_core_news_lg")
        except:
            os.system("python -m spacy download pt_core_news_lg")
            self.nlp = spacy.load("pt_core_news_lg")

    def get_args_rel(self, ext, sent):
        self.alinhamentos = []
        pos = []
        ext_list = ext.split(" ")
        sent_list = sent.split(" ")
        sent_doc = self.nlp(sent)

        # permutando relações
        # Começa com o maior tamanho de subsequência e vai diminuindo
        for length in range(len(ext_list) - 2, 0, -1):
            for start in range(1, len(ext_list) - length):
                end = start + length
                rel = ext_list[start:end]

                idx = (start, end)
                arg0 = " ".join(ext_list[:idx[0]])
                arg1 = " ".join(ext_list[idx[1]:len(sent_list)])
                rel = " ".join(rel)
                valid = self.matcher.match(sent, arg0, rel, arg1)
                if valid[3]:
                    # colhe pos da relação do alinhamento, o pos usado é o da sent nos tokens da ext
                    aux = []
                    aux_dep = []
                    cur_ext = []
                    cur_dep = []
                    for span in valid[:-1]:
                        span_tk = sent_doc[span[0]:span[1] + 1]
                        for token in span_tk:
                            aux.append(token.pos_)
                            aux_dep.append(token.dep_)
                        cur_ext.append(aux)
                        cur_dep.append(aux_dep)
                        aux = []
                        aux_dep = []
                    pos.append(((arg0, rel, arg1), cur_ext, cur_dep))
                    # utiliza regras no pos da relação para filtrar alinhamentos
                    ali_gerado = ((arg0, rel, arg1), cur_ext, cur_dep)
                    rel_pos = ali_gerado[1][1]
                    rel_dep = ali_gerado[2][1]
                    inicio = [[rel_pos[0], rel_dep[0]]]
                    meio = []
                    for x, y in zip(rel_pos[1:-1], rel_dep[1:-1]):
                        meio.append([x, y])
                    fim = [[rel_pos[-1], rel_dep[-1]]]
                    first = False
                    middle = False
                    middle_counter = 0
                    # inicio
                    for i, tags in enumerate(inicio):
                        p_tag = tags[0]
                        p_dep = tags[1]
                        if p_tag == "ADV" and i == 0 and len(rel_pos) > 1 and rel_pos[1] in ['VERB', 'AUX']:
                            first = True
                            if len(rel_pos) == 2:
                                self.alinhamentos.append(ali_gerado[0])
                                return self.alinhamentos
                        elif p_tag == "ADV" and i == 0 and len(rel_pos) > 1 and rel_pos[1] == 'PRON':
                            first = True
                        elif p_tag == "PRON" and i == 0 and len(rel_pos) > 1 and rel_pos[1] in ['VERB', 'AUX']:
                            first = True
                            if len(rel_pos) == 2:
                                self.alinhamentos.append(ali_gerado[0])
                                return self.alinhamentos
                        elif p_tag == "AUX" and i == 0:
                            first = True
                            if len(rel_pos) == 1:
                                self.alinhamentos.append(ali_gerado[0])
                                return self.alinhamentos
                        elif (p_tag == "VERB" and p_dep == "ROOT") and i == 0:
                            first = True
                            if len(rel_pos) == 1:
                                self.alinhamentos.append(ali_gerado[0])
                                return self.alinhamentos
                        elif p_tag == "VERB" and i == 0:
                            first = True
                            if len(rel_pos) == 1:
                                self.alinhamentos.append(ali_gerado[0])
                                return self.alinhamentos

                    # meio
                    for i, tags in enumerate(meio):
                        p_tag = tags[0]
                        if p_tag in ['ADJ', 'NOUN', 'VERB', "AUX", "DET", "PRON", "SCONJ", "PROPN"] and first:
                            middle_counter += 1
                    if middle_counter == len(meio):
                        middle = True
                    # fim
                    for i, tags in enumerate(fim):
                        p_tag = tags[0]
                        if len(rel_pos) == 2 and p_tag == "VERB" and first:
                            self.alinhamentos.append(ali_gerado[0])
                            return self.alinhamentos
                        elif len(rel_pos) == 2 and p_tag == "AUX" and first:
                            self.alinhamentos.append(ali_gerado[0])
                            return self.alinhamentos
                        elif len(rel_pos) == 2 and p_tag == "ADP" and first:
                            self.alinhamentos.append(ali_gerado[0])
                            return self.alinhamentos
                        elif len(rel_pos) > 2 and p_tag in ["ADP", "VERB", "AUX"] and first and middle:
                            self.alinhamentos.append(ali_gerado[0])
                            return self.alinhamentos

        if len(self.alinhamentos) == 0:
            self.alinhamentos.append((" ", " ", " "))

        return self.alinhamentos

class Translators:
    def __init__(self, google: bool):
        if not google:
            #openai.api_key = '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'
            self.prompt_tradução = "Por favor, traduza as seguintes sentenças do inglês para o português. Além disso, identifique e traduza os fatos específicos dentro de cada sentença. Certifique-se de que os fatos traduzidos sejam adaptados para corresponder diretamente à sua representação na sentença traduzida, se baseie nos seguintes exemplos:\n\n" \
            "EXEMPLOS DE ENTRADA E SAÍDA:\n\n" \
            "(entrada):\n" \
            "SENTENÇA: The dog is walking through the park, he is very happy.\n" \
            "FATO: The dog is very happy.\n" \
            "(saida):\n" \
            "SENTENÇA: O cachorro está andando pelo parque, ele está muito feliz.\n" \
            "FATO: O cachorro está muito feliz.\n\n" \
            "(entrada):\n" \
            "SENTENÇA: He made a midnight requisition of all the printers he could lay hands on so that he could monitor all the telephone lines coming into the lab 's computers .\n" \
            "FATO: telephone lines coming the lab 's computers \n" \
            "(saida):\n" \
            "SENTENÇA: Ele fez uma requisição à meia-noite de todas as impressoras que conseguiu encontrar para poder monitorar todas as linhas telefônicas que chegam aos computadores do laboratório.\n" \
            "FATO: linhas telefônicas chegam aos computadores do laboratório.\n\n" \
            "(entrada):\n" \
            "SENTENÇA: The campaign , which started last week and runs through Nov. 23 , with funds earmarked for both the quake and Hugo , `` was Barry 's idea , '' a spokeswoman says .\n" \
            "FATO: The campaign started last week \n" \
            "(saida):\n" \
            "SENTENÇA: A campanha, que começou na semana passada e vai até o dia 23 de novembro, com fundos destinados tanto para o terremoto quanto para o Hugo, 'foi ideia de Barry', disse uma porta-voz.\n" \
            "FATO: A campanha começou na semana passada.\n\n" \
            "(entrada):\n" \
            "SENTENÇA: So far , Nissan 's new - model successes are mostly specialized vehicles with limited sales potential .\n" \
            "FATO: Nissan 's new - model successes specialized limited sales potential \n" \
            "(saida):\n" \
            "SENTENÇA: Até agora, os sucessos dos novos modelos da Nissan são principalmente veículos especializados com potencial de venda limitado.\n" \
            "FATO: Os sucessos dos novos modelos da Nissan são principalmente com potencial de venda limitado.\n"
            #print(self.prompt_tradução)
        else:
            self.google_translator = GoogleTranslator(source="en", target="pt")

    def batch_google(self, txt):
        txt = self.google_translator.translate(txt)
        return txt

    def gpt(self, sent, ext):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=5,
            messages=[
                {"role": "system", "content": self.prompt_tradução},
                {"role": "user", "content": f"SENTENÇA: {sent}"},
                {"role": "user", "content": f"FATO: {ext}"}
            ]
        )
        sentence = response['choices'][0]['message']['content'].split("\n")[0].split(": ")[-1]
        extraction = response['choices'][0]['message']['content'].split("\n")[-1].split(": ")[-1]
        #print("sentence: ", sentence)
        #print("extraction: ", extraction)
        return sentence, extraction

    def gptv2(self, sent, ext):
        url = "http://43.153.203.236:3001/api/chat"
        headers = {
            "content-type": "application/json"
        }
        data = {
            "model": {
                "id": "gpt-3.5-turbo",
                "name": "GPT-3.5",
                "maxLength": 12000,
                "tokenLimit": 3000
            },
            "temperature": 2,
            "messages": [
                {"role": "system",
                 "content": "Você é um tradutor de textos de ingles para portugues brasileiro."},
                {"role": "user", "content": self.prompt_tradução},
                {"role": "user", "content": f"SENTENÇA: {sent}"},
                {"role": "user", "content": f"FATO: {ext}"}
            ]
        }
        response = httpx.post(url, headers=headers, data=json.dumps(data))
        sentence = response.text.split("\n")[0].split(": ")[-1]
        extraction = response.text.split("\n")[-1].split(": ")[-1]
        if len(sentence) == 0 or len(extraction) == 0:
            print("erro na tradução, tentando novamente")
            return self.gptv2(sent, ext)
        return sentence, extraction

    def da_vinci(self, sent, ext):
        pass


class TranslateDataset:
    def __init__(self, dataset_dir: str,
                 dataset_name: str,
                 out_path: str,
                 batch_size: int,
                 google: bool,
                 debug: bool = False
                 ):
        self.batch_size = batch_size
        self.google = google
        self.debug = debug
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.out_path = out_path
        self.translators = Translators(google)
        self.matcher = OIE_Match(sequential=True)
        self.argreleng = ArgsRel()
        self.freezed = []
        self.counter = 0

    def debugging(self, sentence,  ext, raw_sent, raw_ext):
        alignments = self.argreleng.get_args_rel(ext)
        for alignment in alignments:
            arg0_trad = alignment[0]
            rel_trad = alignment[1]
            arg1_trad = alignment[2]
            print("\nDebugging")
            print(f"sent: {sentence}")
            print(f"raw_sent: {raw_sent}")
            print(f"ext: {ext}")
            print(f"raw_ext: {raw_ext}")
            print(f"arg0: {arg0_trad}")
            print(f"rel: {rel_trad}")
            print(f"arg1: {arg1_trad}\n")

    def save_dict(self, data_dict):
        path = self.out_path+"/saida_match"
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        with open(self.out_path+"/saida_match/json_dump.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(data_dict))

    def save_dict_threads(self, n_parts: int):
        data_dict = {}
        for i in range(n_parts):
            with open(f"{self.out_path}/align/data_dict{i}.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                data_dict.update(data)
        self.save_dict(data_dict)

    def save_translate(self, data):
        path = self.out_path+"/translate"
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        with open(self.out_path+"/translate/translate.json", "a", encoding="utf-8") as f:
            open(self.out_path + "/translate/translate.json", "w", encoding="utf-8").close()
            f.write(json.dumps(data))

    def load_dataset(self):
        # estrutura o dataset em um dicionario
        with open(f"{self.out_path}/conll2bioes_output/{self.dataset_name.replace('.conll', '.txt')}",
                  "r", encoding="utf-8") as f:
            data = f.read()
        data = data.split("\n\t")
        data = [ext.split("\n") for ext in data]
        if self.debug:
            data = data[:32]
        for ext in data:
            for i in range(len(ext)):
                ext[i] = ext[i].split("\t")

        dataset = []
        sents = []
        exts = []
        for ext in tqdm(data, desc="Carregando dataset"):
            sentence = ""
            arg0 = ""
            rel = ""
            arg1 = ""
            for e in ext:
                if e != [""]:
                    sentence += e[0] + " "
                    if "ARG0" in e[8]:
                        arg0 += e[0] + " "
                    if "ARG1" in e[8]:
                        arg1 += e[0] + " "
                    if "V" in e[8]:
                        rel += e[0] + " "
            ext = arg0 + rel + arg1
            sents.append(sentence)
            exts.append(ext)
        dataset.append(sents)
        dataset.append(exts)
        return dataset

    def half_translated(self):
        try:
            open(f"{self.out_path}/translate/translate.json", "r", encoding="utf-8")
            return True
        except:
            return False

    def translate_google(self, cache_dir: str):
        cache = Cache(cache_dir)
        dataset = self.load_dataset()


        #traduz dataset
        all_sent = []
        all_ext = []
        raw_sent = []
        raw_ext = []
        for i in tqdm(range(len(dataset[0])), desc=f"Traduzindo dataset"):
            if dataset[0][i] in cache:
                sent = cache[dataset[0][i]]
            else:
                sent = self.translators.batch_google(dataset[0][i])
                cache[dataset[0][i]] = sent
            if dataset[1][i] in cache:
                ext = cache[dataset[1][i]]
            else:
                ext = self.translators.batch_google(dataset[1][i])
                cache[dataset[1][i]] = ext

            all_sent.append(sent)
            all_ext.append(ext)
            raw_sent.append(dataset[0][i])
            raw_ext.append(dataset[1][i])

        cache.clear()
        cache.close()
        trans_dict = {"sent": all_sent, "ext": all_ext, "raw_sent": raw_sent, "raw_ext": raw_ext}
        self.save_translate(trans_dict)

    def translate_gpt(self, dataset=None):
        if dataset is None:
            dataset = self.load_dataset()

        # traduz dataset
        all_sent = []
        all_ext = []
        raw_sent = []
        raw_ext = []

        if self.half_translated():
            with open(f"{self.out_path}/translate/translate.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            all_sent = data["sent"]
            all_ext = data["ext"]
            raw_sent = data["raw_sent"]
            raw_ext = data["raw_ext"]
            i = len(all_sent)
        else:
            i = 0

        while i < len(dataset[0]):
            try:
                sent, ext = self.translators.gptv2(dataset[0][i], dataset[1][i])

                all_sent.append(sent)
                all_ext.append(ext)
                raw_sent.append(dataset[0][i])
                raw_ext.append(dataset[1][i])
                os.system("cls")
                print(f"{i/len(dataset[0])*100:.2f}% concluído ||| {i}/{len(dataset[0])}")
                trans_dict = {"sent": all_sent, "ext": all_ext, "raw_sent": raw_sent, "raw_ext": raw_ext}
                self.save_translate(trans_dict)
                i+=1
            except:
                print("provavelmente o modelo está sobrecarregado, tentando novamente")


        trans_dict = {"sent": all_sent, "ext": all_ext, "raw_sent": raw_sent, "raw_ext": raw_ext}
        self.save_translate(trans_dict)

    def save_translate_thread(self, data, part: int):
        path = self.out_path + f"/translate"
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        with open(self.out_path + f"/translate/translate{part}.json", "a", encoding="utf-8") as f:
            open(self.out_path + f"/translate/translate{part}.json", "w", encoding="utf-8").close()
            f.write(json.dumps(data))

    def half_translated_thread(self, part: int):
        try:
            open(f"{self.out_path}/translate/translate{part}.json", "r", encoding="utf-8")
            return True
        except:
            return False

    def thread_gpt(self, part: int, dataset=None):
        #TODO: dividir em micro_funções
        if dataset is None:
            dataset = self.load_dataset()

        # traduz dataset
        all_sent = []
        all_ext = []
        raw_sent = []
        raw_ext = []

        if self.half_translated_thread(part):
            with open(f"{self.out_path}/translate/translate{part}.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            all_sent = data["sent"]
            all_ext = data["ext"]
            raw_sent = data["raw_sent"]
            raw_ext = data["raw_ext"]
            i = len(all_sent)
        else:
            i = 0

        while i < len(dataset[0]):
            try:
                sent, ext = self.translators.gptv2(dataset[0][i], dataset[1][i])
                if sent == "Error" or ext == "Error":
                    print(f"thread {part} freezou, esperando 30 segundos")
                    self.freezed.append(part)
                    time.sleep(30)
                    print(f"thread {part} liberada")
                    self.freezed.remove(part)
                    raise Exception("Error")

                all_sent.append(sent)
                all_ext.append(ext)
                raw_sent.append(dataset[0][i])
                raw_ext.append(dataset[1][i])
                os.system("cls")
                print(f"{i / len(dataset[0]) * 100:.2f}% concluído ||| {i}/{len(dataset[0])} ||| Thread: {part} ||| Freezed: {self.freezed}")
                trans_dict = {"sent": all_sent, "ext": all_ext, "raw_sent": raw_sent, "raw_ext": raw_ext}
                self.save_translate_thread(trans_dict, part)
                i += 1
            except:
                print("provavelmente o modelo está sobrecarregado, tentando novamente")

        trans_dict = {"sent": all_sent, "ext": all_ext, "raw_sent": raw_sent, "raw_ext": raw_ext}
        self.save_translate_thread(trans_dict, part)

    def merge_translate_parts(self, total_parts:int):
        all_sent = []
        all_ext = []
        raw_sent = []
        raw_ext = []
        with open(self.out_path + f"/translate/translate.json", "a", encoding="utf-8") as f:
            for part in range(total_parts):
                with open(self.out_path + f"/translate/translate{part}.json", "r", encoding="utf-8") as f2:
                    data = json.load(f2)
                    all_sent.extend(data["sent"])
                    all_ext.extend(data["ext"])
                    raw_sent.extend(data["raw_sent"])
                    raw_ext.extend(data["raw_ext"])
            trans_dict = {"sent": all_sent, "ext": all_ext, "raw_sent": raw_sent, "raw_ext": raw_ext}
            f.write(json.dumps(trans_dict))

    def create_dict(self, translate = None, part = None):
        argsRel_eng = ArgsRel3()
        if translate is None:
            with open(self.out_path + "/translate/translate.json", "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = translate
        all_sent = data["sent"]
        all_ext = data["ext"]
        raw_sent = data["raw_sent"]
        raw_ext = data["raw_ext"]
        if self.debug:
            for sent, ext, rs, re in zip(all_sent, all_ext, raw_sent, raw_ext):
                if not self.google:
                    self.debugging(sent, ext, rs, re)
                else:
                    self.debugging(sent, ext, rs, re)
        data_dict = {}
        #identifica elementos da tripla traduzida e armazena em um dicionario
        counter = 0

        for sample in tqdm(zip(all_sent, all_ext), total=len(all_sent)):
            curr_ext = sample[1]
            if curr_ext[-1] == ".":
                curr_ext = curr_ext[:-1]
            alignments = argsRel_eng.get_args_rel(transform_portuguese_contractions(curr_ext), transform_portuguese_contractions(sample[0]))
            for ali in alignments:
                arg0_trad, rel_trad, arg1_trad = ali
                if len(alignments) > 1:
                    match = self.matcher.match(transform_portuguese_contractions(sample[0]),
                                               transform_portuguese_contractions(arg0_trad),
                                               transform_portuguese_contractions(rel_trad),
                                               transform_portuguese_contractions(arg1_trad)
                                               )

                    if match[3] == True:
                        data_dict[str(self.counter)] = {"ID": self.counter,
                                                   "sent": transform_portuguese_contractions(sample[0]),
                                                   "ext": [{"arg1": transform_portuguese_contractions(arg0_trad),
                                                            "rel": transform_portuguese_contractions(rel_trad),
                                                            "arg2": transform_portuguese_contractions(arg1_trad)}]}
                        self.counter += 1
                        break



                else:
                    data_dict[str(self.counter)] = {"ID": self.counter,
                                               "sent": transform_portuguese_contractions(sample[0]),
                                               "ext": [{"arg1": transform_portuguese_contractions(arg0_trad),
                                                        "rel": transform_portuguese_contractions(rel_trad),
                                                        "arg2": transform_portuguese_contractions(arg1_trad)}]}
                    self.counter += 1
            #print(f"{self.counter / (len(all_sent) * 6):.2f}% concluído ||| {self.counter}/{len(all_sent)*6} ||| thread: {part}")

        if part is not None:
            path = self.out_path + f"/align/"
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
            with open(self.out_path + f"/align/data_dict{part}.json", "a", encoding="utf-8") as f:
                f.write(json.dumps(data_dict))
        else:
            #salva dicionario
            self.save_dict(data_dict)

    def create_dict_thread(self, translate = None, part = None):
        argsRel_eng = ArgsRel3()
        if translate is None:
            with open(self.out_path + "/translate/translate.json", "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = translate
        all_sent = data["sent"]
        all_ext = data["ext"]
        raw_sent = data["raw_sent"]
        raw_ext = data["raw_ext"]
        if self.debug:
            for sent, ext, rs, re in zip(all_sent, all_ext, raw_sent, raw_ext):
                if not self.google:
                    self.debugging(sent, ext, rs, re)
                else:
                    self.debugging(sent, ext, rs, re)
        data_dict = {}
        #identifica elementos da tripla traduzida e armazena em um dicionario
        counter = 0

        for sample in zip(all_sent, all_ext):
            curr_ext = sample[1]
            if curr_ext[-1] == ".":
                curr_ext = curr_ext[:-1]
            alignments = argsRel_eng.get_args_rel(transform_portuguese_contractions(curr_ext), transform_portuguese_contractions(sample[0]))
            for ali in alignments:
                arg0_trad, rel_trad, arg1_trad = ali
                if len(alignments) > 1:
                    match = self.matcher.match(transform_portuguese_contractions(sample[0]),
                                               transform_portuguese_contractions(arg0_trad),
                                               transform_portuguese_contractions(rel_trad),
                                               transform_portuguese_contractions(arg1_trad)
                                               )

                    if match[3] == True:
                        data_dict[str(self.counter)] = {"ID": self.counter,
                                                   "sent": transform_portuguese_contractions(sample[0]),
                                                   "ext": [{"arg1": transform_portuguese_contractions(arg0_trad),
                                                            "rel": transform_portuguese_contractions(rel_trad),
                                                            "arg2": transform_portuguese_contractions(arg1_trad)}]}
                        self.counter += 1
                        break



                else:
                    data_dict[str(self.counter)] = {"ID": self.counter,
                                               "sent": transform_portuguese_contractions(sample[0]),
                                               "ext": [{"arg1": transform_portuguese_contractions(arg0_trad),
                                                        "rel": transform_portuguese_contractions(rel_trad),
                                                        "arg2": transform_portuguese_contractions(arg1_trad)}]}
                    self.counter += 1
            print(f"{(self.counter / (len(all_sent) * 6))*100:.2f}% concluído ||| {self.counter}/{len(all_sent)*6} ||| thread: {part}")

        if part is not None:
            path = self.out_path + f"/align/"
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
            with open(self.out_path + f"/align/data_dict{part}.json", "a", encoding="utf-8") as f:
                f.write(json.dumps(data_dict))
        else:
            #salva dicionario
            self.save_dict(data_dict)



def run(batch_size: int,
        dataset_dir: str,
        dataset_name: str,
        test_size: float,
        dev_size: float,
        translated: bool,
        debug: bool = False,
        use_google: bool = True,
        sequential: bool = True,
        cache_dir: str = "cache"
        ):
    converted = True
    OUT_NAME = dataset_name.replace(".conll", "")
    INPUT_PATH = ""

    path = "outputs"+"/"+OUT_NAME
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    json_dir = path+"/saida_match"
    pathlib.Path(json_dir).mkdir(parents=True, exist_ok=True)

    if use_google or debug:
        batch_size = 1
    trans_eng = TranslateDataset(dataset_dir, dataset_name, path, debug=debug, batch_size=batch_size, google=use_google)
    if translated:
        pass
    else:
        if use_google:
            LoadDataset(dataset_dir, dataset_name, path)
            print("Traduzindo com Google")
            trans_eng.translate_google(cache_dir=cache_dir)
        else:
            LoadDataset(dataset_dir, dataset_name, path)
            print("Traduzindo com ChatGPT")
            trans_eng.translate_gpt()
    trans_eng.create_dict()
    criar_conll(OUT_NAME, INPUT_PATH, test_size, dev_size, converted=converted, sequential=sequential)
