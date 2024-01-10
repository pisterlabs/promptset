from bs4 import BeautifulSoup
import math
import jsonlines
import requests
import traceback
from multiprocessing import Pool, cpu_count
import os, re
import xml.etree.ElementTree as etree
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import openai

TOKEN = "U4DMV*8nvpm3EOpvf69Rxw(("

STEM = [
    "stackoverflow",  # always do first since it's the largest
    "serverfault", "superuser", "webapps", "gaming", "webmasters", "gamedev", "stats", "math",
      "tex", "askubuntu",
    "unix", "wordpress", "cstheory", "electronics", "android", "physics", "dba", "scifi",
    "codereview", "codegolf", "quant", "drupal", "sharepoint", "sqa", "crypto", "dsp", "bitcoin", "linguistics",
    "scicomp", "biology", "mathematica", "cs", "chemistry", "raspberrypi", "patents", "genealogy",
    "robotics", "expressionengine", "reverseengineering", "networkengineering", "opendata", "mathoverflow", "space",
    "sound", "astronomy", "tor", "ham", "arduino", "joomla", "datascience", "craftcms", "emacs", "economics",
    "engineering", "civicrm", "opensource", "elementaryos", "computergraphics", "hardwarerecs",
    "3dprinting", "ethereum", "retrocomputing", "monero", "ai", "sitecore", "iot", "devops", "bioinformatics",
    "cseducators", "iota", "stellar", "conlang", "quantumcomputing", "eosio", "tezos", "drones", "materials",
    "cardano", "proofassistants", "substrate", "bioacoustics", "solana",
]

OTHER = [
    "cooking", "photo", "diy", "superuser", "gis", "money", "english", "stackapps", "ux", "apple", "rpg",
    "bicycles", "boardgames", "homebrew", "security", "writers", "avp", "graphicdesign", "pm", "skeptics", "fitness",
    "mechanics", "parenting", "music", "judaism", "german", "japanese", "philosophy", "gardening", "travel", "french",
    "christianity", "hermeneutics", "history", "bricks", "spanish", "movies", "chinese", "poker", "outdoors",
    "martialarts", "sports", "academia", "workplace", "chess", "russian", "islam", "salesforce", "politics", "anime",
    "magento", "ell", "sustainability", "tridion", "freelancing", "blender", "italian", "pt", "aviation",
    "ebooks", "beer", "softwarerecs", "expatriates", "matheducators", "earthscience", "puzzling", "buddhism",
    "moderators", "worldbuilding", "ja", "hsm", "lifehacks", "coffee", "vi", "musicfans", "woodworking", "ru",
    "rus", "mythology", "law", "portuguese", "es", "latin", "languagelearning", "crafts", "korean", "esperanto",
    "literature", "vegetarianism", "ukrainian", "interpersonal", "or",
]

NICHE = []

ALL = STEM + OTHER + NICHE

# TODO fix the split between the sets.
# assert len(STEM) + len(OTHER) + len(NICHE) == 179, "Total sites should be 179, not " + str(len(STEM) + len(OTHER) + len(NICHE))
# assert len(STEM) == 75, "STEM sites should be 75, not " + str(len(STEM))
# assert len(OTHER) == 99, "OTHER sites should be 99, not " + str(len(OTHER))
# assert len(NICHE) == 5, "NICHE sites should be 5, not " + str(len(NICHE))
print("Stem Size", len(STEM))
print("Other Size", len(OTHER))
print("Niche Size", len(NICHE))

class Stack_Exchange_Downloader:
    def __init__(self, name):
        """
        :param name: name of stackexchange site to download.
        If all, will download all stackexchanges & metas.
        """
        sitesmap = requests.get(
            "https://ia600107.us.archive.org/27/items/stackexchange/Sites.xml"
        ).content
        self.name = (
            name.replace("http://", "")
            .replace("https://", "")
            .replace(".com", "")
            .replace(".net", "")
        )
        self.sites = {}
        self.parse_sitesmap(sitesmap)

    def parse_sitesmap(self, sitesmap):
        soup = BeautifulSoup(sitesmap, "lxml")
        for site in soup.find_all("row"):
            url = site["url"].replace("https://", "")
            site_name = site["tinyname"]
            download_link = "https://archive.org/download/stackexchange/" + url + ".7z"
            if url == "stackoverflow.com":
                download_link = "https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z"
            self.sites[site_name] = {"url": url, "download": download_link}

    def download(self):
        dl_list = self.sites if self.name == "all" else {self.name: self.sites[self.name]}
        for k in dl_list:
            command = f"wget {dl_list[k]['download']} -P dumps"
            print(command)
            if os.system(command):
                raise Exception(f"Download for {k} failed!")

    def extract(self):
        extract_list = (
            self.sites if self.name == "all" else {self.name: self.sites[self.name]}
        )
        for k, site in extract_list.items():
            command = "py7zr x dumps/{} dumps/{}".format(
                site["download"].replace(
                    "https://archive.org/download/stackexchange/", ""
                ),
                k,
            )
            print(command)
            if os.system(command):
                raise Exception(f"Extraction for {k} failed!")

def header_info(xml_path):
    os.system("head {}".format(xml_path))


def handle_unicode_errors(txt):
    return txt.encode('utf-8', 'replace').decode()


def is_question(elem_attribs):
    if elem_attribs["PostTypeId"] is not None:
        if elem_attribs["PostTypeId"] == "1":
            return True
    return False


def is_answer(elem_attribs):
    if elem_attribs["PostTypeId"] is not None:
        if elem_attribs["PostTypeId"] == "2":
            return True
    return False


def filter_newlines(text):
    return re.sub("\n{3,}", "\n\n", text)


def is_accepted_answer(a_attribs, q_attribs):
    assert is_question(q_attribs), "Must be a question to have an accepted answer"
    assert is_answer(a_attribs), "Must be an answer to be an accepted answer"
    if q_attribs["AcceptedAnswerId"] is not None:
        if q_attribs["AcceptedAnswerId"] == a_attribs["Id"]:
            return True
    else:
        return False


def has_answers(elem_attribs):
    assert is_question(elem_attribs), "Must be a question to have answers"
    if elem_attribs["AnswerCount"] is not None:
        if int(elem_attribs["AnswerCount"]):
            return True
    return False


def trim_attribs(elem_attribs, attrib_type="question"):
    """deletes non-useful data from attribs dict for questions / answers, returns remaining"""
    if attrib_type == "question":
        to_keep = ['Id', 'Body', 'Title', 'Tags', 'AnswerCount', 'AcceptedAnswerId', 'PostTypeId', 'Score']
        to_delete = [x for x in elem_attribs.keys() if x not in to_keep]
        [elem_attribs.pop(x, None) for x in to_delete]
        elem_attribs["ParsedAnswers"] = 0
        elem_attribs["Answers"] = {}
    elif attrib_type == "answer":
        to_keep = ['Id', 'Body', 'Score']
        new_dict = {}
        for item in to_keep:
            new_dict[item] = elem_attribs[item]
        return new_dict
    else:
        raise Exception('Unrecognized attribute type - please specify either question or answer')

class QA_Pairer():

    def __init__(self, xml_path, name=None, out_folder="out"):
        """Makes a text dataset from StackExchange dumps"""
        self.xml_path = xml_path
        if name is None:
            self.name = os.path.dirname(xml_path).replace("dumps/", "")
        else:
            self.name = name
        # dict to save questions
        self.questions = defaultdict(lambda: None, {})
        # folder to save txt files to
        self.out_folder = out_folder
        self.output_buffer = []

    def questions_count(self):
        return sum(1 for _, elem in tqdm(etree.iterparse(self.xml_path, events=('end',)), desc="Parsing {} XML file".format(self.name)) if elem.tag == "row" and is_question(elem.attrib) or (elem.clear() and False))
    def main(self):
        """iterates through SE xmls and:

        - stores PostTypeId="1" with AcceptedAnswerIds / Answers.
        - when an AcceptedAnswerId or Answer > min_score is reached, it should:
            > concat the Question & Accepted answer
            > Clean markup / HTML
            > Output to txt file
            > Delete from memory

        """
        os.makedirs(self.out_folder, exist_ok=True)
        for event, elem in tqdm(etree.iterparse(self.xml_path, events=('end',)), desc="Parsing {} XML file".format(self.name)):
            if len(self.output_buffer) > 1000:
                break # we'll do some manual filtering from here to get down to our actual dataset.
            if elem.tag == "row":
                try:
                    attribs = defaultdict(lambda: None, elem.attrib)
                    if is_question(attribs):
                        if has_answers(attribs):
                            trim_attribs(attribs, "question")
                            self.questions[attribs["Id"]] = attribs
                        else:
                            # if the question has no answers, discard it
                            continue
                    elif is_answer(attribs):
                        self.add_answer(attribs)
                        self.check_complete(attribs)
                    elem.clear() # saves memory... these are big XML files
                except:
                    traceback.print_exc()
        output = sorted(self.output_buffer, key=lambda x: x['question_score'], reverse=True)
        jsonlines.open(f"json-out/{self.name}.jsonl", mode='w').write_all(output)

    def add_answer(self, a_attribs):
        """
        Adds answer to its parent question in self.questions if it's either an accepted answer or above self.min_score.
         If answer is an accepted answer, it gets appended to the AcceptedAnswer field, otherwise it gets appended to
         OtherAnswers.

         Also increments the question's 'ParsedAnswers' field. When ParsedAnswers = AnswerCount, the question is deleted
         from memory and saved to a text file.

        :param a_attribs: Answer's attribute dict
        """
        assert is_answer(a_attribs), "Must be an answer to add to parent"
        if a_attribs is not None and self.questions[a_attribs["ParentId"]] is not None:
            if is_accepted_answer(a_attribs, self.questions[a_attribs["ParentId"]]):
                self.questions[a_attribs["ParentId"]]["Answers"][a_attribs["Id"]] = trim_attribs(a_attribs, "answer")
                self.questions[a_attribs["ParentId"]]["ParsedAnswers"] += 1
            else:
                # TODO why might an answer not have a score?
                assert "Score" in a_attribs and a_attribs["Score"]
                # score = int(a_attribs["Score"]) if a_attribs["Score"] is not None else 0
                if a_attribs["Id"] is not None:
                    parent = self.questions[a_attribs["ParentId"]]
                    if parent is not None:
                        self.questions[a_attribs["ParentId"]]["Answers"][a_attribs["Id"]] = trim_attribs(a_attribs, "answer")
                        self.questions[a_attribs["ParentId"]]["ParsedAnswers"] += 1
                else:
                    # TODO why would an answer not have an ID?
                    self.questions[a_attribs["ParentId"]]["ParsedAnswers"] += 1

    def check_complete(self, a_attribs):
        """
        checks if the parent question of the previously added answer has no future answers, and if so,
        removes from dict and prints to file.
        """
        parent = self.questions[a_attribs["ParentId"]]
        if a_attribs is not None and parent is not None:
            assert "Score" in parent, parent
            if parent["AnswerCount"] is not None and parent["ParsedAnswers"] is not None:
                if int(parent["ParsedAnswers"]) == int(parent['AnswerCount']):
                    self.questions.pop(a_attribs["ParentId"], None)
                    if parent["Answers"] is not None and len(parent["Answers"]) > 0:
                        if parent["Title"] is not None:
                            title_str = BeautifulSoup(parent["Title"], "lxml").get_text()
                        if parent["Body"] is not None:
                            body_str = BeautifulSoup(parent["Body"], "lxml").get_text()
                        if parent["Answers"] is not None:
                            ans_obj = max(parent["Answers"].values(), key=lambda item: int(item["Score"]))
                            ans = BeautifulSoup(ans_obj["Body"], "lxml").get_text()
                            ans_score = int(ans_obj["Score"])
                            if len(ans) < 1200 or len(ans) > 4096 or ans_score < 10:
                                return
                        try:
                            self.output_buffer.append({
                                'title': filter_newlines(title_str) if parent["Title"] is not None else "",
                                'description': filter_newlines(body_str) if parent["Body"] is not None else "",
                                'answer': filter_newlines(ans) if parent["Answers"] is not None else "",
                                'question_score': parent["Score"],
                                'answer_score': ans_score
                            })
                        except:
                            self.output_buffer.append({
                                'title': filter_newlines(handle_unicode_errors(BeautifulSoup(parent["Title"], "html.parser").get_text())) if parent["Title"] is not None else "",
                                'description': filter_newlines(handle_unicode_errors(BeautifulSoup(parent["Body"], "html.parser").get_text())) if parent["Body"] is not None else "",
                                'answer': filter_newlines(handle_unicode_errors(ans)) if parent["Answers"] is not None else "",
                                'question_score': parent["Score"],
                                'answer_score': ans_score
                            })

def download_and_process_single(name):
    name = name.strip().lower()
    os.makedirs("dumps", exist_ok=True)
    s = Stack_Exchange_Downloader(name)
    # assert set(ALL).issubset(set(s.sites.keys()))
    path_to_xml = f"dumps/{name}/Posts.xml"
    if name != "stackoverflow":
        path_to_7z = f"dumps/{s.sites[name]['url']}.7z"
    else:
        path_to_7z = "dumps/stackoverflow.com-Posts.7z"
    out_folder = f"out/{name}"
    os.makedirs(out_folder, exist_ok=True)
    if not os.path.isfile(path_to_xml):
        if not os.path.isfile(path_to_7z):
            s.download()  # download 7z if it's not downloaded already
        s.extract()  # extract 7z if it's not extracted already
        try:
            os.remove(path_to_7z)
        except FileNotFoundError:
            print(
                "ERROR: FileNotFoundError: File {} not found".format(
                    s.sites[name]["url"]
                )
            )
    qa = QA_Pairer(path_to_xml, out_folder=out_folder, name=name)
    qa.main()

    # for f in os.listdir(f"dumps/{name}"):
    #     if f.endswith(".xml"):
    #         os.remove(os.path.join(f"dumps/{name}", f))

def cnt(name):
    name = name.strip().lower()
    # assert set(ALL).issubset(set(s.sites.keys()))
    path_to_xml = f"dumps/{name}/Posts.xml"
    qa = QA_Pairer(path_to_xml, out_folder="blahblahblah", name=name)
    return qa.questions_count()

SMALL_TEST = False
if __name__ == "__main__":
#   if SMALL_TEST:
#     download_and_process_single("webapps") # useful for testing to run this and comment the below
#   else:
#     cpu_no = cpu_count() - 1
#     p = Pool(cpu_no)
#     p.map(download_and_process_single, OTHER)

    # r = Pool(47).map(cnt, STEM)

    # how do we do sampling here? softmax will just push the biggest community to 1.0...
    # print(softmax(r, temperature=3.0))

    # this sampling does not follow the paper. not sure how they did it...
    # r = np.array(r)
    # desired_counts = list((r / r.sum() * 200).astype(np.int32))
    # ord = sorted(zip(range(len(desired_counts)), desired_counts), key=lambda item:item[1])
    # for i in range(200 - sum(desired_counts)):
    #     desired_counts[ord[i % len(ord)][0]] += 1
    # plan = zip(STEM, desired_counts)
    # print(list(plan))
    # plan = [('cooking', 1), ('photo', 1), ('diy', 4), ('superuser', 29), ('gis', 9), ('money', 2), ('english', 7), ('stackapps', 1), ('ux', 1), ('apple', 7), ('rpg', 2), ('bicycles', 1), ('boardgames', 1), ('homebrew', 1), ('security', 4), ('writers', 1), ('avp', 1), ('graphicdesign', 2), ('pm', 1), ('skeptics', 1), ('fitness', 1), ('mechanics', 1), ('parenting', 1), ('music', 1), ('judaism', 2), ('german', 1), ('japanese', 1), ('philosophy', 1), ('gardening', 1), ('travel', 2), ('french', 1), ('christianity', 1), ('hermeneutics', 1), ('history', 1), ('bricks', 1), ('spanish', 1), ('movies', 1), ('chinese', 1), ('poker', 1), ('outdoors', 1), ('martialarts', 1), ('sports', 1), ('academia', 2), ('workplace', 1), ('chess', 1), ('russian', 1), ('islam', 1), ('salesforce', 7), ('politics', 1), ('anime', 1), ('magento', 6), ('ell', 6), ('sustainability', 1), ('tridion', 1), ('freelancing', 1), ('blender', 6), ('italian', 1), ('pt', 9), ('aviation', 1), ('ebooks', 1), ('beer', 1), ('softwarerecs', 1), ('expatriates', 1), ('matheducators', 1), ('earthscience', 1), ('puzzling', 1), ('buddhism', 1), ('moderators', 1), ('worldbuilding', 2), ('ja', 1), ('hsm', 1), ('lifehacks', 0), ('coffee', 0), ('vi', 0), ('musicfans', 0), ('woodworking', 0), ('ru', 26), ('rus', 1), ('mythology', 0), ('law', 1), ('portuguese', 0), ('es', 11), ('latin', 0), ('languagelearning', 0), ('crafts', 0), ('korean', 0), ('esperanto', 0), ('literature', 0), ('vegetarianism', 0), ('ukrainian', 0), ('interpersonal', 0), ('or', 0)]
    plan = [('stackoverflow', 161), ('serverfault', 2), ('superuser', 3), ('webapps', 1), ('gaming', 1), ('webmasters', 1), ('gamedev', 1), ('stats', 1), ('math', 10), ('tex', 1), ('askubuntu', 2), ('unix', 1), ('wordpress', 1), ('cstheory', 1), ('electronics', 1), ('android', 1), ('physics', 1), ('dba', 1), ('scifi', 1), ('codereview', 1), ('codegolf', 1), ('quant', 1), ('drupal', 1), ('sharepoint', 1), ('sqa', 1), ('crypto', 1), ('dsp', 1), ('bitcoin', 0), ('linguistics', 0), ('scicomp', 0), ('biology', 0), ('mathematica', 0), ('cs', 0), ('chemistry', 0), ('raspberrypi', 0), ('patents', 0), ('genealogy', 0), ('robotics', 0), ('expressionengine', 0), ('reverseengineering', 0), ('networkengineering', 0), ('opendata', 0), ('mathoverflow', 0), ('space', 0), ('sound', 0), ('astronomy', 0), ('tor', 0), ('ham', 0), ('arduino', 0), ('joomla', 0), ('datascience', 0), ('craftcms', 0), ('emacs', 0), ('economics', 0), ('engineering', 0), ('civicrm', 0), ('opensource', 0), ('elementaryos', 0), ('computergraphics', 0), ('hardwarerecs', 0), ('3dprinting', 0), ('ethereum', 0), ('retrocomputing', 0), ('monero', 0), ('ai', 0), ('sitecore', 0), ('iot', 0), ('devops', 0), ('bioinformatics', 0), ('cseducators', 0), ('iota', 0), ('stellar', 0), ('conlang', 0), ('quantumcomputing', 0), ('eosio', 0), ('tezos', 0), ('drones', 0), ('materials', 0), ('cardano', 0), ('proofassistants', 0), ('substrate', 0), ('bioacoustics', 0), ('solana', 0)]
    global_output = []
    # openai.api_key = 
    for category, desired_cnt in plan:
        local_output = []
        print("loading file", category)
        with open(f"json-out/{category}.jsonl", "r") as f:
            print("file loaded", category)
            while len(local_output) < desired_cnt:
                blob = f.readline()
                def run():
                    return openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        temperature=0.0,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": blob+"\n\nDo not consider the title or description. Is the answer written in first person? Respond with the word \"yes\" or the word \"no\""},
                        ]
                    )
                try:
                    print("attempt1")
                    completion = run()
                except:
                    try:
                        print("attempt2")
                        completion = run()
                    except:
                        print("attempt3")
                        try:
                            completion = run()
                        except openai.error.RateLimitError as e:
                            from time import sleep
                            print("sleeping 60 seconds since we hit rate limit error")
                            sleep(60)
                            try:
                                print("attempt4")
                                completion = run()
                            except:
                                print("attempt5")
                                completion = run()
                if 'no' in completion['choices'][0]['message']['content'].lower():
                    local_output.append(blob)
                    print(blob)
                    print("global status pos", len(global_output) + len(local_output))
                else:
                    print("written in first person so skipped", completion['choices'][0]['message']['content'].lower())
        global_output += local_output
    with open(f"stem.jsonl", "w") as f:
        f.write("".join(global_output))
