import os
import re
from typing import List, Tuple

import openai
openai.api_key = os.environ["OPENAI_API_KEY"]


def _remove_strings(text: str) -> str:
    # text = re.sub(r'"[^"]*?"', '""', text)   
    # text = re.sub(r"'[^']*?'", "''", text)
    text = re.sub(r'(["\']).*?\1', '""', text)
    return text


class CodeReviewer:
    def __init__(self, verbose: bool = False) -> None:
        self._verbose = verbose
        self._model = "gpt-4"
        self._model_params = {
            # "max_tokens": 4096,
            "temperature": 0.9,
        }
        self._system_message = {
            "role": "system",
            "content": (
                "You are a Senior Software Engineer. Your main task is to review code "
                "written by both junior and senior developers. You always try to be "
                "as helpful as possible, but you also want to make sure that the code "
                "is of high quality. You are super expert in Python and you just provide "
                "code review for Python code. Whenever you see some non-Python code, you "
                "simply ignore it.\n Python code is showed you with the following format:\n\n"
                "<start_file_name> The name of the file <end_file_name>\n"
                "<start_code> The code contained in the file <end_code>\n\n"
                "When you review the code you provide multiple comments. "
                "Each comment must have the following format:\n\n"
                "<start_file_name> The name of the file your comment "
                "refers to <end_file_name>\n"
                "<start_code_snippet> Code snippet <end_code_snippet>\n"
                "<start_comment> Your comment <end_comment>\n\n"
                "Note that a code snippet is usually just a small piece of the full code. "
                "You can also provide multiple comments for the same code snippet.\n\n"
                "When writing comments you must follow few simple rules:\n"
                "1. if you do not have any comment on the code just write 'LGTM!'. "
                "You should write it just when you have NO comment at all.\n"
                "2. you MUST write the code in the snippet section in the "
                "exact same way it is written in the original code. Consider that "
                "the snippet you provide will be used for retrieve its exact "
                "position with a regex later. Please when possible return just one "
                "line in the snippet.\n"
                "3. you really care about the quality of the code and you hate to see "
                "some anti-patterns. You also want to make sure that the code is "
                "readable and easy to understand. You also want to make sure that "
                "the code is well tested and that the tests are easy to understand.\n\n"
                "Please consider that you will not see the full code in a single text. "
                "You will receive one "
                "file .py at a time. You will then provide your comments for that file. We "
                "will then send you the next file. You will provide your comments for that "
                "file. We will then send you the next file. And so on. So don't be surprised "
                "if you don't see the tests at the beginning.\nWhenever you realize that you " 
                "need to comment a previous file, you just need to put the filename you want "
                "to refer to into the <start_file_name><end_file_name> section when you "
                "write the comment.\n"
            ),
        }
        self._messages = [self._system_message]
    
    def reset(self) -> None:
        self._messages = [self._system_message]
    
    @staticmethod
    def _process_model_message(
            model_response: str, 
            user_message: dict, 
            old_messages: filter,
    ) -> List[Tuple[str, int, str]]:
        code = user_message["content"]
        original_file_name = code.split("<start_file_name>")[1].split("<end_file_name>")[0].strip()
        cleaned_code = re.search(r'<start_code>(.*)<end_code>', code, re.DOTALL)
        if cleaned_code is None:
            raise ValueError(f"Code not found for message: {user_message}")
        cleaned_code = cleaned_code.group(1).strip()
        comments = [
            comment.strip() 
            for comment in model_response.split("<start_file_name>") 
            if len(comment.strip()) > 0
        ]
        processed_comments = []
        for comment in comments:
            file_name = comment.split("<end_file_name>")[0].strip()
            cleaned_comment = re.search(r'<start_comment>(.*?)<end_comment>', comment, re.DOTALL)
            if cleaned_comment is None:
                print(f"WARNING: comment not found for comment: {comment}")
                continue
            cleaned_comment = cleaned_comment.group(1).strip()    
            code_snippet = re.search(r'<start_code_snippet>(.*?)<end_code_snippet>', comment, re.DOTALL)
            if code_snippet is None:
                print(f"WARNING: code snippet not found for comment: {comment}")
                continue
            code_snippet = code_snippet.group(1).strip()
            if file_name == original_file_name:
                index = _remove_strings(cleaned_code.split(code_snippet)[0]).strip().count("\n") + 1
            else:
                index = 1  # when not found, we assume that the comment is for the first line
                for previous_code in old_messages:
                    previous_code = previous_code["content"]
                    previous_file_name = previous_code.split("<start_file_name>")[1].split("<end_file_name>")[0].strip()
                    cleaned_previous_code = re.search(r'<start_code>(.*)<end_code>', previous_code, re.DOTALL)
                    if cleaned_previous_code is None:
                        continue
                    cleaned_previous_code = cleaned_previous_code.group(1).strip()
                    if previous_file_name == file_name:
                        index = _remove_strings(cleaned_previous_code.split(code_snippet)[0]).strip().count("\n") + 1
                        break
            processed_comments.append((file_name, index, cleaned_comment))
        return processed_comments
        
    
    def __call__(self, filename: str, code: str) -> List[Tuple[str, int, str]]:
        if code is None or len(code) == 0:
            return []
        code = f"<start_file_name> {filename} <end_file_name>\n<start_code> {code} <end_code>"
        user_message = {
            "role": "user",
            "content": code,
        }
        self._messages.append(user_message)
        if self._verbose:
            print(f"OpenAI request: {self._messages}")
        response = openai.ChatCompletion.create(
            model=self._model,
            messages=self._messages,
            **self._model_params,
        )
        model_response = response["choices"][0]["message"]["content"]
        if self._verbose:
            print(f"OpenAI response: {response}")
        comments = self._process_model_message(
            model_response, 
            self._messages[-1], 
            filter(lambda x: x["role"] == "user", self._messages)
        )
        if self._verbose:
            print(f"Comments: {comments}")
        model_message = {
            "role": "assistant",
            "content": model_response,
        }
        self._messages.append(model_message)
        return comments


if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("--filename", type=str, required=True)
    args = parser.parse_args()
    
    with open(args.filename, "r") as f:
        code = f.read()
        filename = args.filename.split("/")[-1]
    
    comments = CodeReviewer(verbose=True)(filename, code)
    print(comments)