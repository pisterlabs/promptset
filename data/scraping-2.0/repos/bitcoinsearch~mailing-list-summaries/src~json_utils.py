import openai
from xml.etree import ElementTree as ET
from datetime import datetime
import os
import json
import nltk
import pytz
from loguru import logger

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

from src.utils import preprocess_email, clean_title, get_id, month_dict
from src.gpt_utils import create_n_bullets, create_summary, generate_chatgpt_summary_for_prompt


class GenerateJSON:

    def get_xml_summary(self, data):
        number = get_id(data["_source"]["id"])
        title = data["_source"]["title"]
        xml_name = clean_title(title)
        published_at = datetime.strptime(data['_source']['created_at'], '%Y-%m-%dT%H:%M:%S.%fZ')
        published_at = pytz.UTC.localize(published_at)
        month_name = month_dict[int(published_at.month)]
        str_month_year = f"{month_name}_{int(published_at.year)}"
        dev_name = data['_source']['dev_name']

        current_directory = os.getcwd()
        file_path = f"static/{dev_name}/{str_month_year}/{number}_{xml_name}.xml"
        full_path = os.path.join(current_directory, file_path)

        if os.path.exists(full_path):
            namespaces = {'atom': 'http://www.w3.org/2005/Atom'}
            tree = ET.parse(full_path)
            root = tree.getroot()
            summ_list = root.findall(".//atom:entry/atom:summary", namespaces)
            summ = "\n".join([summ.text for summ in summ_list])
            author_list = root.findall(".//atom:author", namespaces)
            author_ = "\n".join([a.find('atom:name', namespaces).text for a in author_list])
            author_ = " ".join(author_.split(" ")[:-2])
            return f"{author_}:{summ}\n"
        else:
            logger.warning(f"No xml file found: {full_path}")
            return ""

    def generate_recent_posts_summary(self, dict_list):
        logger.info("working on given post's summary")

        recent_post_data = ""

        for data in dict_list:
            xml_summ = self.get_xml_summary(data)
            recent_post_data += xml_summ

            if xml_summ is None:
                body = data['_source']['body']
                author_ = data['_source']['authors']
                author_ = ", ".join([a for a in author_])
                body = preprocess_email(body)
                body_summ = create_summary(body)
                summ = f"{author_}:{body_summ}\n"
                recent_post_data += summ
        recent_post_data = create_summary(recent_post_data)

        summ_prompt = f"""You are required to produce a concise header summary from a compilation of condensed recent discussions. Transform the following extracted text from mailing lists into a brief summary composed of only three or four significant sentences, adhering to these important criteria:
    Guidelines:
        1. While synthesizing, refrain from or reword phrases like "The context discusses...", "The email discusses...", "In this context...", "The context covers...", "The context questions...", "In this email...", "The email covers..." and similar phrases.
        2. The summarization must have a formal tone and be high in informational content.
        3. Ensure that punctuation is followed by a space and that all syntax rules are adhered to.
        4. Any links given within the text should be retained and appropriately incorporated.
        5. Rather than being a simple rewording of the original content, the summary should restructure and simplify the main points.
        6. Mention full names (both the first name and last name) of the authors if applicable. 
        7. Break down the summary into concise, meaningful paragraphs ensuring each paragraph captures a unique aspect or perspective from the original text, provided it should be no longer than three or four sentences.
        8. Please ensure that the summary does not start with labels like "Email 1:", "Email 2:" and so on.
        \n CONTEXT:\n\n{recent_post_data}"""

        response_str = generate_chatgpt_summary_for_prompt(summarization_prompt=summ_prompt, max_tokens=500)
        return response_str

    def create_single_entry(self, data, base_url_for_xml="static", look_for_combined_summary=False,
                            remove_xml_extension=False, add_combined_summary_field=False):
        number = get_id(data["_source"]["id"])
        title = data["_source"]["title"]
        published_at = datetime.strptime(data['_source']['created_at'], '%Y-%m-%dT%H:%M:%S.%fZ')
        published_at = pytz.UTC.localize(published_at)
        contributors = data['_source']['contributors']
        url = data['_source']['url']
        authors = data['_source']['authors']
        body = data['_source']['body']
        local_dev_name = data['_source']['dev_name']
        xml_name = clean_title(title)
        month_name = month_dict[int(published_at.month)]
        str_month_year = f"{month_name}_{int(published_at.year)}"
        current_directory = os.getcwd()

        combined_summ_file_path = ""
        base_path = f"{base_url_for_xml}/{local_dev_name}/{str_month_year}"
        file_extension = "" if remove_xml_extension else ".xml"
        if look_for_combined_summary:
            if os.path.exists(f"static/{local_dev_name}/{str_month_year}/combined_{xml_name}.xml"):
                combined_summ_file_path = f"{base_path}/combined_{xml_name}{file_extension}"
            else:
                combined_summ_file_path = ""
        file_path = f"{base_path}/{number}_{xml_name}{file_extension}"

        # fetch the summary from xml if exist
        xml_summary = self.get_xml_summary(data)

        if xml_summary is None:
            xml_summary = create_summary(body)

        bullets = create_n_bullets(xml_summary, n=3)

        entry_data = {
            "id": number,
            "title": title,
            "link": url,
            "authors": authors,
            "published_at": published_at.isoformat(),
            "summary": bullets,
            "n_threads": data["_source"]["n_threads"],
            "dev_name": local_dev_name,
            "contributors": contributors,
            "file_path": file_path,
            "combined_summ_file_path": combined_summ_file_path
        }

        combined_summ_bullets = ""
        if add_combined_summary_field:
            combined_summ_full_path = os.path.join(current_directory, combined_summ_file_path)

            if os.path.exists(combined_summ_full_path):
                namespaces = {'atom': 'http://www.w3.org/2005/Atom'}
                tree = ET.parse(combined_summ_full_path)
                root = tree.getroot()
                summ_list = root.findall(".//atom:entry/atom:summary", namespaces)
                combined_summ = "\n".join([summ.text for summ in summ_list])
                combined_summ_bullets = create_n_bullets(combined_summ, n=3)

            entry_data['combined_summary'] = combined_summ_bullets if combined_summ_bullets else ""

        return entry_data

    def get_existing_json_ids(self, file_path):
        current_directory = os.getcwd()
        full_path = os.path.join(current_directory, file_path)
        if os.path.exists(full_path):
            with open(full_path, 'r') as j:
                data = json.load(j)
            id_list = [item['title'] for item in data['recent_posts']]
            id_list = id_list + [item['title'] for item in data['active_posts']]
            return id_list
        else:
            logger.warning(f"No existing homepage.json file found: {full_path}")
            return []

    def is_body_text_long(self, data, sent_threshold=2):
        body_text = data['_source']['body']
        body_text = preprocess_email(body_text)
        body_token = sent_tokenize(body_text)
        logger.info(f"Body sentence token length: {len(body_token)}")
        return len(body_token) > sent_threshold
