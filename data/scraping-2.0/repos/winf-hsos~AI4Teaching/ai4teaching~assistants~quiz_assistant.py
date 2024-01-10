from ai4teaching import Assistant
from ai4teaching import log
from openai import OpenAI
import os
import json
import random

class QuizAssistant(Assistant):
    def __init__(self, config_file, depending_on_assistant=None):
        log("Initializing QuizAssistant")
        log(f"QuizAssistant depends on {depending_on_assistant}") if depending_on_assistant else None
        super().__init__(config_file, depending_on_assistant)

        self.openai_client = OpenAI()
        self.openai_model = "gpt-4-1106-preview"
        self.question_json_file = self.config["question_json_file"] if "question_json_file" in self.config else None

    def create_multiple_choice_question(self, topic, retrieval=False, n_answers=4):
        if retrieval == True:
            retrieved_documents = self.vector_db.query(topic, n_results=1)
            chunk_ids = retrieved_documents["ids"][0]
            question = self._create_question_with_retrieval(
                chunk_ids[0], n_answers=n_answers
            )
        else:
            question = self._create_question_without_retrieval(
                topic, n_answers=n_answers
            )

        question_json = json.loads(question)
        formatted_question = self._format_mc_question_object(question_json)
        return formatted_question

    def create_multiple_choice_question_for_text(self, text, n_answers=4):
        question = self._create_question_from_text(text, n_answers=n_answers)
        question_json = json.loads(question)
        formatted_question = self._format_mc_question_object(question_json)
        return formatted_question    

    def save_question(self, question):
        data = self._check_and_load_saved_questions()
        data.append(question)

        # Save file
        with open(self.question_json_file, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False)

    def update_saved_question(self, index, updated_question):
        data = self._check_and_load_saved_questions()
        data[index] = updated_question
            
        # Save file
        with open(self.question_json_file, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False)

    def delete_saved_question(self, index):
        data = self._check_and_load_saved_questions()
        del data[index]

        with open(self.question_json_file, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False)

    def get_saved_questions(self):
        return self._check_and_load_saved_questions()

    def load_random_question_from_file(self):
        if self.question_json_file is None:
            log("No >question_json_file< specified in config file.", type="warning")
            return

        # Check if file exists
        if not os.path.isfile(self.question_json_file):
            log(f"File {self.question_json_file} does not exist. No questionns added yet?", type="warning")
            return

        # Read current content from file
        with open(self.question_json_file, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
            random_question = random.choice(data)
            return random_question

    def export_questions_to_kahoot(self, file_path):
        # TODO: Implement
        ...

    def _check_and_load_saved_questions(self):
        if self.question_json_file is None:
            log("No >question_json_file< specified in config file.", type="warning")
            return None

        # Check if file exists
        if not os.path.isfile(self.question_json_file):
            log(f"File {self.question_json_file} does not exist. No questionns added yet?", type="warning")
            return []

        # Read current content from file
        with open(self.question_json_file, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
            return data

    def _create_question_from_text(self, text, n_answers=4):
        messages = []
        messages.append(
            {
                "role": "system",
                "content": "Create a question with four answer options. The question and answers must be in German. Please use UTF-8 encoding. Only one of the options is correct, all others are false. The question should be about a text passage the user provides in the prompt. Never make a reference to the text in the question. Do end your question with \"... that is mentioned in the text passage\" or similar. Ask with no mention of the text.",
            }
        )
        messages.append({ "role": "user", "content": f"Here is text: \"{text}\"" })

        mc_question = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=messages,
            tools=[{"type": "function", "function": self._create_function_json()}],
        )

        return mc_question.choices[0].message.tool_calls[0].function.arguments

    def _create_question_with_retrieval(self, chunk_id, n_answers=4):
        print(chunk_id)

    def _create_question_without_retrieval(self, topic, n_answers=4):
        messages = []
        messages.append(
            {
                "role": "system",
                "content": "Create a question with four answer options. The question and answers must be in German. Please use UTF-8 encoding. Only one of the options is correct, all others are false. The question should be about a topic the user provides in the prompt.",
            }
        )
        messages.append({ "role": "user", "content": f"The topic: {topic}" })

        mc_question = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=messages,
            tools=[{"type": "function", "function": self._create_function_json()}],
        )

        # TODO: Check if result is successful
        return mc_question.choices[0].message.tool_calls[0].function.arguments

    def _format_mc_question_object(self, question_json):
        question = {}
        question["question_text"] = question_json["question_text"]

        answer_options = []
        question["answer_options"] = answer_options

        for key in question_json:
            if "answer_option" in key:
                answer_option = {}
                answer_option["text"] = question_json[key]

                if key.startswith("correct"):
                    answer_option["is_correct"] = True
                else:
                    answer_option["is_correct"] = False
                answer_options.append(answer_option)
        
        # Randomize order of answer options
        random.shuffle(answer_options)        

        return question

    def _create_function_json(self, n_answers=4):
        return {
            "name": "create_multiple_choice_question",
            "description": "Creates a multiple choice question with four answer options, only one is correct.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question_text": {
                        "type": "string",
                        "description": "The text of the question",
                    },
                    "wrong_answer_option_a": {
                        "type": "string",
                        "description": "Wrong answer option A.",
                    },
                    "wrong_answer_option_b": {
                        "type": "string",
                        "description": "Wrong answer option B.",
                    },
                    "wrong_answer_option_c": {
                        "type": "string",
                        "description": "Wrong answer option C.",
                    },
                    "correct_answer_option_d": {
                        "type": "string",
                        "description": "The only correct answer option D.",
                    },
                },
                "required": [
                    "question_text",
                    "wrong_answer_option_a",
                    "wrong_answer_option_b",
                    "wrong_answer_option_c",
                    "correct_answer_option_d",
                ],
            },
        }
