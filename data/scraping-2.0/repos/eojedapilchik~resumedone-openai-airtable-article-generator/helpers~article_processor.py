import os
import datetime
from typing import List, Dict
from models.article import Article
from models.prompt_command_factory import PromptCommandFactory
from helpers.openai_handler import OpenAIException


class ArticleProcessor:
    def __init__(self, openai_handler, record_id, airtable_handler):
        self.openai_handler = openai_handler
        self.airtable_handler = airtable_handler
        self.record_id = record_id
        self.metadata = {"meta title": "", "meta description": ""}
        self.images_url = []
        self.log_text = ""
        self.retries = int(os.getenv("OPENAI_RETRIES", 3))

    def process(self, prompts: List[Dict], article: Article) -> List[Dict]:
        for index, prompt in enumerate(prompts, start=1):
            prompt_type = prompt.get("type", "").lower().strip()
            command = PromptCommandFactory.create_command(prompt_type)
            try:
                print(f"Prompt {index}/{len(prompts)} in progress...")
                command.execute(prompt, self.retries, article=article, openai_handler=self.openai_handler,
                                airtable_handler=self.airtable_handler)
                self.log_text += f"\n -- [PROMPT {index}]:  {prompt.get('prompt')}.\n"
                print(f"(x) Prompt {index} of {len(prompts)} completed successfully.")
                self.update_airtable_record_log(self.record_id,
                                                new_status=f'Prompt {index} of {len(prompts)} completed successfully.')
            except OpenAIException as e:
                continue
            except Exception as e:
                print(f"[!!] Error executing command: {str(e)}")
                continue

        return prompts

    def update_airtable_record_log(self, record_id, new_status: str = 'Error'):
        try:
            fields = {
                "fldpnyajPwaBXM6Zb": new_status,
            }
            self.airtable_handler.update_record(record_id, fields)
        except Exception as e:
            print(f"[!!] Error updating Airtable log for record: {str(e)}")

    def update_airtable_record(self, record_id: str, prompts: List[Dict],
                               elapsed_time_bf_at: float = 0):
        print("[+] Updating Airtable record...")
        if len(prompts) <= 0:
            print("[-] Insufficient responses provided.")
            return None
        # loop through prompts and extract responses
        responses = []
        metadata = {}
        samples = {}
        for prompt in prompts:
            if prompt.get("response"):
                responses.append(prompt.get("response"))
            if prompt.get("metadata"):
                metadata[prompt["metadata"]["type"]] = prompt["metadata"]["value"]
            if prompt.get("sample_cover_letter"):
                samples["sample_cover_letter"] = prompt.get("sample_cover_letter")
        plain_text_responses = [response["plain_text"] for response in prompts]
        current_utc_time = datetime.datetime.utcnow()
        iso8601_date = current_utc_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        article_html = ''.join(responses)
        article_html = article_html.replace('""', '\'')

        try:
            fields = {
                "fld7vn74uF0ZxQhXe": article_html,
                "fldus7pUQ61eM1ymY": elapsed_time_bf_at,
                "fldsnne20dP9s0nUz": "To Review",
                "fldTk3wrPUWrx0AjP": iso8601_date,
                "fldLgR2ao2astuLbs": ''.join(plain_text_responses),
                "fldpnyajPwaBXM6Zb": self.log_text if self.log_text != "" else "Success",
                "fldIvmfoPfkJbYDcy": metadata.get("meta description"),
                "fld4v3esUgKDDH9aI": metadata.get("meta title"),
                "fldK0YcZgm3gV4I2l": samples.get("sample_cover_letter") if samples.get("sample_cover_letter") else "",
            }
            self.airtable_handler.update_record(record_id, fields)
            print("[+] Airtable record updated successfully.")
        except Exception as e:
            print(f"[!!] Error updating record: {str(e)}")
            print(prompts)
