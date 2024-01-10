import openai
import os
import tiktoken
import json
import csv


class ChatTemplate:
    @classmethod
    def get_templates(cls, csv_path):
        chat_template = ChatTemplate(csv_path)
        return chat_template.templates

    def __init__(self, csv_path) -> None:
        self.csv_path = csv_path
        self._templates = self._read_chat_templates()

    def _read_chat_templates(self) -> list[dict[str, str]]:
        with open(self.csv_path) as csv_file:
            templates = csv.DictReader(csv_file)

            return [template for template in templates]

    @property
    def templates(self) -> list[dict[str, str]]:
        if len(self._templates) < 1:
            raise IndexError("templatesの中身が空です")

        return self._templates


class TrainingJsonFormatter:
    input_csv: str
    output_dir: str

    def __init__(self, input_csv, output_dir) -> None:
        self.input_csv = input_csv
        self.output_dir = output_dir
        self.file_name = ""

        if not self.output_dir:
            self.output_dir = os.path.dirname(os.path.abspath(__file__))

    def create_format(self):
        templates = ChatTemplate.get_templates(self.input_csv)
        file = self.input_csv.split("/")[-1]
        self.file_name = file.split(".")[0]
        data = [
            {
                "messages": [
                    templates[0],
                    templates[i],
                    templates[i + 1],
                ]
            }
            for i in range(1, len(templates), 2)
        ]

        return data

    def saved_train_file(self):
        messages = self.create_format()
        saved_file_path = os.path.join(self.output_dir, f"{self.file_name}.jsonl")
        for message in messages:
            with open(saved_file_path, 'a', encoding='utf-8') as f:
                json_line = json.dumps(message, ensure_ascii=False)
                f.write(json_line + "\n")

