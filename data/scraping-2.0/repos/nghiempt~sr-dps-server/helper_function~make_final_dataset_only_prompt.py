import asyncio
import ssl
from urllib.request import urlopen
from bs4 import BeautifulSoup
import csv
import json
import openai
import os
from dotenv import load_dotenv
from newspaper import Article


class READ_DATA_SAFETY:

    @staticmethod
    def load_html_content(url):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        html = urlopen(url, context=ctx).read()
        soup = BeautifulSoup(html, "html.parser")
        return soup

    @staticmethod
    def indent_content(tag):
        tag_content = tag.get_text(separator='\n')
        if tag_content.strip() == '· Optional':
            return str(tag.contents[0])
        if (tag_content.strip() != '· Optional') & (tag.name != 'h3'):
            return '\n data-section: ' + str(tag.contents[0])
        tag_name = tag.name
        if tag_name == 'h3':
            return '\n\t category: ' + str(tag.contents[0])
        elif tag_name == 'h2':
            return str(tag.contents[0]) + '\n'
        else:
            return ''

    @staticmethod
    def filter_content(soup):
        global title
        title = soup.find('div', class_='ylijCc').get_text(separator='\n')
        content = ""
        tags = soup.find_all(['h2', 'h3', 'h4'])
        div_tag = soup.find_all('div', class_='FnWDne')
        i = 0
        j = 0
        for i in range(len(tags)):
            if tags[i].contents:
                span_tag = tags[i].find('span')
                if span_tag:
                    span_text = span_tag.get_text(separator='\n')
                    if span_text.strip() != '· Optional':
                        content += '\n\t\tdata-type: ' + span_text
                        content += '\n\t\tpurpose: ' + \
                            div_tag[j].get_text(
                                separator='\n')
                        j += 1
                    span_tag.decompose()
                if tags[i].contents:
                    content += READ_DATA_SAFETY.indent_content(tags[i])
        return content

    @staticmethod
    async def scrape_link(url):
        loop = asyncio.get_event_loop()
        soup = await loop.run_in_executor(None, READ_DATA_SAFETY.load_html_content, url)
        text = READ_DATA_SAFETY.filter_content(soup)
        return text

    @staticmethod
    def formated_data_v0(content):
    #     content_data = ''
    #     lines = content.splitlines()
    #     for i in range(len(lines)):
    #         category = data_section = data_type = purpose = optional = ''
    #         if lines[i].startswith(' data-section:') and not lines[i].startswith(' data-section: Security practices'):
    #             data_section = lines[i].replace(' data-section: ', '')
    #             content_data += data_section + ": "
    #             for j in range(i + 1, len(lines)):
    #                 if lines[j].startswith('\t category: '):
    #                     category = lines[j].replace('\t category: ', '')
    #                     content_data = content_data + category+"("
    #                     for k in range(j + 1, len(lines)):
    #                         if lines[k].startswith('\t\tdata-type: '):
    #                             data_type = lines[k].replace(
    #                                 '\t\tdata-type: ', '')
    #                             content_data = content_data + data_type+", "
    #                         elif lines[k].startswith('\t\tpurpose: '):
    #                             purpose = lines[k].replace('\t\tpurpose: ', '')
    #                             if purpose.endswith(' · Optional'):
    #                                 optional = True
    #                                 purpose = purpose.replace(
    #                                     ' · Optional', '')
    #                             else:
    #                                 optional = False
    #                             purpose = purpose.replace(', ', ' - ')

    #                         elif lines[j].startswith('\t category: '):
    #                             content_data = content_data[:-2] + "), "
    #                             break
    #                 elif lines[j].startswith(' data-section:'):
    #                     content_data = content_data[:-2] + "; "
    #                     break
    #         elif lines[i].startswith(' data-section: Security practices'):
    #             data_section = lines[i].replace(' data-section: ', '')
    #             content_data += data_section + ": "
    #             for j in range(i + 1, len(lines)):
    #                 if lines[j].startswith('\t category: '):
    #                     category = lines[j].replace('\t category: ', '')
    #                     content_data = content_data + category + ", "
    #                 elif lines[j].startswith(' data-section:'):
    #                     content_data = content_data[:-2] + "; "
    #                     break
    #     content_data = content_data[:-2]
        # sections = content_data.split(';')
        # result_dict = {}
        # for section in sections:
        #     parts = section.split(':')
        #     key = parts[0].strip()
        #     value = parts[1].strip() if len(parts) > 1 else None
        #     result_dict[key] = value

        return ""

    @staticmethod
    def formated_data(content):
        result_json = {}

        lines = content.splitlines()
        current_section = None

        for line in lines:
            if line.startswith(' data-section: No data shared'):
                current_section = "data_shared"
                result_json[current_section] = []
            elif line.startswith(' data-section: No data collected'):
                current_section = "data_collected"
                result_json[current_section] = []
            elif line.startswith(' data-section:'):
                current_section = line.replace(' data-section: ', '').strip()
                current_section = current_section.replace(' ', '_').lower()
                result_json[current_section] = []
            elif line.startswith('\t category: '):
                category = line.replace('\t category: ', '').strip()
                current_data = {"category": category, "sub_info": []}

                for sub_line in lines[lines.index(line) + 1:]:
                    if sub_line.startswith('\t\tdata-type: '):
                        data_type = sub_line.replace('\t\tdata-type: ', '').strip()
                    elif sub_line.startswith('\t\tpurpose: '):
                        purpose = sub_line.replace('\t\tpurpose: ', '').strip()
                        optional = purpose.endswith(' · Optional')
                        purpose = purpose.replace(' · Optional', '').replace(', ', ' - ')
                        current_data["sub_info"].append({
                            "data_type": data_type,
                            "purpose": purpose,
                            "optional": optional
                        })
                    elif sub_line.startswith('\t category: '):
                        break

                result_json[current_section].append(current_data)
        return result_json

    @staticmethod
    def formated_data_string_only(content):
        content_data = ''
        lines = content.splitlines()
        for i in range(len(lines)):
            category = data_section = data_type = purpose = ''
            if lines[i].startswith(' data-section: No data shared'):
                data_section = lines[i].replace(' data-section: ', '')
                content_data += "[Data shared: " + data_section + "] - "
            elif lines[i].startswith(' data-section: No data collected'):
                data_section = lines[i].replace(' data-section: ', '')
                content_data += "[Data collected: " + data_section + "] - "
            elif lines[i].startswith(' data-section:') and not lines[i].startswith(' data-section: Security practices'):
                data_section = lines[i].replace(' data-section: ', '')
                content_data += "[" + data_section + ": "
                for j in range(i + 1, len(lines)):
                    if lines[j].startswith('\t category: '):
                        category = lines[j].replace('\t category: ', '')
                        content_data = content_data + category + "("
                        for k in range(j + 1, len(lines)):
                            if lines[k].startswith('\t\tdata-type: '):
                                data_type = lines[k].replace('\t\tdata-type: ', '')
                                content_data = content_data + data_type + " · <"
                            elif lines[k].startswith('\t\tpurpose: '):
                                purpose = lines[k].replace('\t\tpurpose: ', '').replace(", ", " - ")
                                if purpose.endswith(' · Optional'):
                                    purpose = purpose.replace(" · Optional", "")
                                    content_data += purpose + "> · Optional, "
                                else:
                                    content_data += purpose + ">, "
                            elif lines[j].startswith('\t category: '):
                                content_data = content_data[:-2] + "), "
                                break
                    elif lines[j].startswith(' data-section:'):
                        content_data = content_data[:-2] + "] - "
                        break
            elif lines[i].startswith(' data-section: Security practices'):
                data_section = lines[i].replace(' data-section: ', '')
                content_data += "[" + data_section + ": "
                for j in range(i + 1, len(lines)):
                    if lines[j].startswith('\t category: '):
                        category = lines[j].replace('\t category: ', '')
                        content_data = content_data + category + ", "
                content_data = content_data[:-2] + "] - "
        content_data = content_data[:-3]
        return content_data

    @staticmethod
    def generate_result(user_url):
        preprocess_datasafety = asyncio.run(
            READ_DATA_SAFETY().scrape_link(user_url))
        content_data_safety = READ_DATA_SAFETY().formated_data(
            preprocess_datasafety)
        return content_data_safety


class READ_PRIVACY_POLICY:

    def __init__(self):
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")

    @staticmethod
    def get_completion(prompt, model="gpt-4"):
        messages = [{"role": "user", "content": prompt}]
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.9,
        )
        return response.choices[0].message.content

    @staticmethod
    def remove_empty_lines(content):
        lines = content.split('\n')
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        return '\n'.join(cleaned_lines)

    @staticmethod
    def check_valid_token_prompt(prompt):
        print(len(prompt))
        return len(prompt) <= 8000

    @staticmethod
    def generate_result(url):
        try:
            article = Article(url)
            article.download()
            article.parse()
            text = READ_PRIVACY_POLICY.remove_empty_lines(article.text)
            prompt = 'Help me to find the origin text about 3 things: type/purpose of data the app shared with others, type/purpose of data the app collected and Security Practices in the text below in this json format: {"data_shared" : "a string", "data_collected": "a string", "security_practices" : "a string"} . Please in the answer, just give me the json only and in English: \n'
            check_valid_token_prompt = READ_PRIVACY_POLICY.check_valid_token_prompt(
                prompt + text)
            if check_valid_token_prompt:
                response = READ_PRIVACY_POLICY.get_completion(prompt + text)
                return response
            else:
                return "No provide sharing information section"
        except Exception as e:
            print(f"An exception occurred: {e}")
            return "An error occurred during processing"
        
    @staticmethod
    def generate_result_string_only(url):
        try:
            article = Article(url)
            article.download()
            article.parse()
            text = READ_PRIVACY_POLICY.remove_empty_lines(article.text)
            prompt = 'Help me to find the origin text about 3 things: type/purpose of data the app shared with others, type/purpose of data the app collected and Security Practices in the text below in this json format: {"data_shared" : "a string", "data_collected": "a string", "security_practices" : "a string"} . Please in the answer, just give me the json only and in English: \n'
            check_valid_token_prompt = READ_PRIVACY_POLICY.check_valid_token_prompt(
                prompt + text)
            if check_valid_token_prompt:
                response = READ_PRIVACY_POLICY.get_completion(prompt + text)
                if response.startswith("{"):
                    data_dict = json.loads(response)
                    data_shared = data_dict["data_shared"]
                    data_collected = data_dict["data_collected"]
                    security_practices = data_dict["security_practices"]
                    formatted_output = f"[Data shared: {data_shared}] - [Data Collected: {data_collected}] - [Security practices: {security_practices}]"
                    print(formatted_output)
                    return formatted_output
                else:
                    return response
            else:
                return "No provide sharing information section"
        except Exception as e:
            print(f"An exception occurred: {e}")
            return "An error occurred during processing"



class MAKE_PROMPT:

    @staticmethod
    def get_prompt(id, ds_content, pp_content):
        json_prompt = '''
            Let's compare and analyze the information between Data Safety and Privacy Policy to clarify 3 issues: which information is incorrect, which information is incomplete and which information is inconsistent.\n\nNotes when classifying:\n+ Incomplete: Data Safety provides information but is not as complete as the Privacy Policy provides.\n+ Incorrect: Data Safety does not provide that information, but the Privacy Policy mentions it.\n+ Inconsistency: Data Safety is provided but its description is inconsistent with the Privacy Policy information provided.\n\nNote: always gives me the result (0 or 1) in the form below:\nIncomplete: 0 or 1 (1 is yes, 0 is no)\nIncorrect: 0 or 1 (1 is yes, 0 is no)\nInconsistency: 0 or 1 (1 is yes, 0 is no)\n\nBelow is information for 2 parts:
            Data Safety - Share section: ''' + ds_content + '''
            Privacy Policy - Share section: ''' + pp_content + '''
            '''
        return json_prompt


class JSON_MAKER:

    @staticmethod
    def remove_empty_lines(content):
        lines = content.split('\n')
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        return '\n'.join(cleaned_lines)

    @staticmethod
    def get_dataset(prompt, completion, package_name, json_file_path):
        with open(json_file_path, 'r') as json_file:
            existing_data = json.load(json_file)
        newdata = {
            "prompt": JSON_MAKER().remove_empty_lines(prompt),
            "completion": JSON_MAKER().remove_empty_lines(completion),
            "package_name": package_name,
        }
        existing_data.append(newdata)
        with open(json_file_path, 'w') as json_file:
            json.dump(existing_data, json_file, indent=4)

    @staticmethod
    def loop_csv(csv_path, json_path, step_5, step_6, step_7):
        with open(csv_path, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for index, row in enumerate(reader):
                print("\n_____________ Run times " +
                      row[0] + " <" + row[2] + "> " + "_____________")
                content_5 = step_5.generate_result(row[7])
                content_6 = step_6.generate_result(row[8])
                content_7 = step_7.get_prompt(row[0], content_5, content_6)
                content_8 = ""
                JSON_MAKER.get_dataset(content_7, content_8, row[2], json_path)
                print("~~~~~~~~~~~~~~ Success ~~~~~~~~~~~~~~\n")


if __name__ == "__main__":
    step_5 = READ_DATA_SAFETY()
    step_6 = READ_PRIVACY_POLICY()
    step_7 = MAKE_PROMPT()

    csv_path = ""
    json_path = ""

    JSON_MAKER().loop_csv(csv_path, json_path, step_5, step_6, step_7)
