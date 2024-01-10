import re
import glob
import json
import time
import faiss
import numpy as np
import unicodedata
from dotenv import load_dotenv

from tqdm import tqdm
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
import uuid
from langchain.docstore.in_memory import InMemoryDocstore
from dataclasses import dataclass, fields
from datetime import datetime

# ADD TO CONTENT TO REMOVE \XA STUFF
# norm_i = unicodedata.normalize("NFKD", i)
# title += f" {norm_i}"

env_config = load_dotenv()  # create .env file with your OPENAI_API_KEY


@dataclass
class ParsedDocument:
    act_name: str
    full_act_name: str
    section_number: str
    section_name: str
    filter_tag: str
    content: str
    url: str
    full_section: str

    def __hash__(self):
        return hash(self.content)


def delete_refs(input_string):
    result = re.sub(r'\[.*?\]', '', input_string)
    return result


def split_into_subsections(section, url, section_number):
    section = delete_refs(section)
    subsections = section.split(".")
    # print("!!!!!!!!!!!!!!!!!!!!!!")
    # print(section)
    # print("-----------------")
    # for x in subsections:
    #    print(x)
    # Connect cases, e.g 'of the Companies Act (Cap. 50), as the case may'
    # e.g. sub sub sections started with (b) and '.' was before
    connected_subsections = []
    meta_subsection_number = {}
    for i in range(len(subsections)):
        if i == 0:
            connected_subsections.append(subsections[i])
            continue

        s = subsections[i].strip()

        if "Deleted by" in s:
            continue

        if not (s.startswith("[") or s.startswith("(")):
            # fix problems with header before number as in section 12 of https://sso.agc.gov.sg/Act-Rev/ITA1947/Published/20211231?DocDate=20211231&ProvIds=P13-#P13-
            try:
                j_s = s.replace(" ", "")
                i1 = j_s.index("(")
                i2 = j_s.index(")")
                if j_s[i2 + 1].isupper() and j_s[i1 + 1].isnumeric():
                    ss = j_s[i1 + 1:i2]
                    meta_subsection_number[s] = f"({ss})"
                    # print("-------------")
                    # print(ss)
                    # print(section_number, url)
                    # print("Upper: ", s)
                    # print([x[:10] for x in connected_subsections])
                    # print("-------------")
                    connected_subsections.append(s)
                    continue
            except:
                pass

            old_sub = connected_subsections[-1]
            connected_subsections[-1] = connected_subsections[-1] + "." + subsections[i]
            if old_sub in meta_subsection_number:
                ss = meta_subsection_number.pop(old_sub)
                meta_subsection_number[connected_subsections[-1]] = ss
        elif s.startswith("(") and s.replace(" ", "")[1].isalpha():
            connected_subsections[-1] = connected_subsections[-1] + "." + subsections[i]
        else:
            connected_subsections.append(subsections[i])

    subsections = {}
    for x in connected_subsections:
        if x in meta_subsection_number:
            subsections[meta_subsection_number[x]] = x
            continue
        ss = ''
        try:
            if x.strip().index("(") < 10:
                x = x[x.strip().index("("):]
        except ValueError as err:
            subsections[ss] = x
            continue

        if x.strip().startswith('('):
            ss = x[:x.index(")") + 1].strip()

        if ss in subsections and ss:
            # print("new ss: ", ss)
            # print("new x: x")
            # input()
            subsections[ss] = subsections[ss] + x
        subsections[ss] = x

    return subsections


def create_doc_filter_tag(doc_title):
    return re.sub(r"\d+", "", doc_title).replace(" ", "")


def get_latest_urls(documents, fps):
    docs_name_to_url = dict()
    for doc, fp in zip(documents, fps):
        title = re.sub(r'[0-9]+', '', doc["title"]).strip(" ")
        # title = doc["title"]
        if title not in docs_name_to_url:
            docs_name_to_url[title] = doc["url"]
        elif docs_name_to_url[title] != doc["url"]:
            try:
                if "/Act/" in doc["url"] and "/Act-Rev/" in docs_name_to_url[title]:
                    docs_name_to_url[title] = doc["url"]
                    continue
                print("doc['url']:", doc["url"])
                if "DocDate=" in doc["url"]:
                    str_date = doc["url"].split("DocDate=")[1][:8]
                else:
                    str_date = ""
                date_obj_new = datetime.strptime(str_date, "%Y%m%d")
                date_obj_in_dict = datetime.strptime(docs_name_to_url[title].split("DocDate=")[1][:8], "%Y%m%d")
                if date_obj_new > date_obj_in_dict:
                    docs_name_to_url[title] = doc["url"]
            except Exception as err:
                print(docs_name_to_url[title], doc["url"])
                print(fp)
                print(err)
                input()

    return list(docs_name_to_url.values()), docs_name_to_url


def parse_unstructured(body_data, url, short_title, act_name, regex_pattern=r"^\d+[A-Z]*\.$"):
    part_name, paragraph_name = '', ''
    start = None
    parsed_documents = []
    ignore_indices = []
    for ind, line in enumerate(body_data):
        pattern_matched = re.match(regex_pattern, line[:10])
        if line.lower().startswith("part"):
            part_name = 'Part name: '
            inds_to_check = [i for i in range(1, 5) if len(body_data) > ind + i]
            for k in inds_to_check:
                if body_data[ind + k].isupper() and not re.search(regex_pattern, body_data[ind + k][:10]):
                    ignore_indices.append(ind + k)
                    part_name += body_data[ind + k]
        if pattern_matched:
            if start is not None:
                # paragraph_name = f"#{section_number[:-1]} " + paragraph_name
                # part_pref = part_name + ". " if part_name else ""
                # structured[part_pref + paragraph_name] =
                content = ' '.join([
                    body_data[li] for li in range(start, ind - 1)
                    if li not in ignore_indices
                ])

                full_section = unicodedata.normalize("NFKD", content)

                subsections = split_into_subsections(full_section, url, section_number)

                content = f"{short_title}. {paragraph_name}. " + full_section

                parsed_documents.append(
                    ParsedDocument(
                        act_name=short_title,
                        full_act_name=act_name,
                        section_number=section_number[:-1],
                        section_name=paragraph_name,
                        filter_tag=create_doc_filter_tag(short_title),
                        content=content,
                        url=url,
                        full_section=full_section
                    )
                )

            for j in range(ind - 1, -1, -1):
                if re.search(regex_pattern, body_data[j][:10]):
                    continue
                else:
                    ignore_indices.append(j)
                    paragraph_name = body_data[j]
                    break

            start = ind + 1
            section_number = pattern_matched.group(0)
    return parsed_documents


t1 = time.time()

data_files_path = "data/combined/"
files = glob.glob(f"{data_files_path}*.json")
print(files)
all_parsed_documents = {}
url_to_revised_year = {}
skipped = []

latest_urls = []

all_data = []
fps = []
for file_path in files:
    if "Supplement" in file_path:
        continue  # ignore this docs as they are a description of what were changed in the main act
    with open(file_path, "r") as f:
        data = json.load(f)
        all_data.extend(data)
        fps.extend([file_path] * len(data))

latest_urls, urls_as_dict = get_latest_urls(all_data, fps)
with open("latest_urls.json", "w") as f:
    json.dump(urls_as_dict, f)

for file_path in tqdm(files):
    with open(file_path, "r") as f:
        data = json.load(f)

    for document in data:
        if document["url"] not in latest_urls:
            continue

        short_title = document["title"]

        # with open("check_json.json", "w") as f:
        #     json.dump(document, f, indent=2)

        if "notification" in short_title.lower() or "amendment" in document["title"].lower() or "bill" in document[
            "title"].lower():
            skipped.append(short_title)
            continue

        # print(document)
        # input()
        # print("================")
        title = ''
        if "body_raw" not in document:
            continue

        ### REMOVE THIS, ADDED ONLY FOR TESTS
        # if "order" in document["title"].lower():
        #     continue

        if "front" in document:
            for i in document[
                "front"]:  # add docs when act name is same but it is some Chapter of the doc, e.g. Income Tax Act and Income Tax Act Chapter 134
                # if i.lower().startswith("(chapter"):
                norm_i = unicodedata.normalize("NFKD", i)
                title += f" {norm_i}"

            for x in document["front"]:
                if "REVISED EDITION" in x and "Original Enactment" not in x:
                    try:
                        revised_year = int(x.replace("REVISED EDITION", ""))
                    except Exception as err:
                        print(document["url"])
                        print(document["front"])
                        print(" ")
        else:
            title = short_title
        revised_year = 0
        for x in document["front"]:
            if "REVISED EDITION" in x and "Original Enactment" not in x:
                try:
                    revised_year = int(x.replace("REVISED EDITION", ""))
                except Exception as err:
                    print(document["url"])
                    print(document["front"])
                    print(" ")

        url_to_revised_year[document["url"]] = revised_year
        parsed_documents = parse_unstructured(document["body_raw"], document["url"], short_title, title)

        for x in parsed_documents:
            key = f"{title} {x.section_number} {x.section_name}"
            # #if x.section_name == "Trading operations carried on partly in Singapore":
            # print(x.section_number, x.subsection_number)
            # print(x.content)
            # print(x.url)
            #     pass
            all_parsed_documents[key] = x

# print(set(skipped))
# for k, v in all_parsed_documents.items():
#     if v.act_name.startswith("Income Tax"):
#         print(v.act_name, v.url, v.section_number)

print(f"Len docs: {len(all_parsed_documents)}")
input()

texts = list(map(lambda x: x.content, all_parsed_documents.values()))
model_kwargs = {'device': 'cpu'}

embeddings = HuggingFaceEmbeddings(
    model_kwargs=model_kwargs
)

embs = embeddings.embed_documents(texts)

# from transformers import AutoTokenizer, AutoModel
# from torch import Tensor
# from tqdm import tqdm
# import torch

# def average_pool(last_hidden_states: Tensor,
#                  attention_mask: Tensor) -> Tensor:
#     last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
#     return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-large')
# model = AutoModel.from_pretrained('intfloat/e5-large')
# model.cuda()
# model.eval()

# embeddings = []
# for x in tqdm(all_parsed_documents.values()):
#     embed_text = 'passage: ' + x.content
#     batch_dict = tokenizer([embed_text], padding=True, truncation=True, return_tensors='pt').to("cuda:0")

#     outputs = model(**batch_dict)
#     embedding = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
#     embeddings.append(embedding.detach().cpu())
#     torch.cuda.empty_cache()

# embs = torch.cat(embeddings).numpy()


# Build index manualy
# section_names = list(map(lambda x: x.section_name, all_parsed_documents.values()))


# print(f"len embs: {embs}")
embs = np.array(embs, dtype=np.float32)

index = faiss.IndexFlatIP(embs.shape[1])
faiss.normalize_L2(embs)
index.add(embs)

# Prepare data for FAISS init
# metadatas = None
metadatas = []
for parsed_doc in all_parsed_documents.values():
    m_doc = {}
    for field in fields(parsed_doc):
        if field.name != "content":
            m_doc[field.name] = getattr(parsed_doc, field.name)
    metadatas.append(m_doc)

documents = []
for i, text in enumerate(texts):
    metadata = metadatas[i] if metadatas else {}
    documents.append(Document(page_content=text, metadata=metadata))
index_to_id = {i: str(uuid.uuid4()) for i in range(len(documents))}
docstore = InMemoryDocstore(
    {index_to_id[i]: doc for i, doc in enumerate(documents)}
)

db = FAISS(None, index, docstore, index_to_id, )

save_path = "ft_embeddings"
# save_path = "full_acts_index_cos_sim_fixed_keys_add_metadata_as_dict_add_v2_filter_tag_fixed_names"

db.save_local(save_path)

print(f"Time elapsed: {(time.time() - t1) / 60}")
