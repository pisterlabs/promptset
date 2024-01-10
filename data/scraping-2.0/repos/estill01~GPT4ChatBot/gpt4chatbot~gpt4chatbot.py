import openai
import os
import time
import json
import shutil
from pathlib import Path
from threading import Thread
import math

# TODO Refactor - extract functionality to properly organized file structure (it's time..)
# TODO Integrate ChromaDB ; can query here first and then if no hits ping GPT4 API
# TODO Implement summary-based 'context stretching'
# TODO Get it to work with git repos ; make commits and etc like git2gpt does

class GPT4ChatBot:
    def __init__(self, api_key, auto_summarize_interval=5, auto_higher_order_summarize_base=5):
        # TODO swap this for .. ChromaDB(?)
        self.chat_dir = "chats"
        Path(self.chat_dir).mkdir(exist_ok=True)

        self.active_chat = None
        self.chats = {}
        openai.api_key = api_key
        self.stream_callback = self._default_stream_callback
        self.auto_summarize_interval = auto_summarize_interval
        self.auto_higher_order_summarize_interval = auto_higher_order_summarize_interval
        self.auto_higher_order_summarize_base = auto_higher_order_summarize_base
        self.working_on_project = None


    def create_chat(self, prompt=None, description=None, generate_description=True):
        if prompt is None:
            prompt = "This is the start of a new chat."

        if description is None:
            if generate_description:
                description = self._generate_description(initial_prompt)

        chat_id = str(time.time()).replace(".", "")

        chat_path = os.path.join(self.chat_dir, chat_id)
        os.makedirs(chat_path, exist_ok=True)

        with open(os.path.join(chat_path, "log.txt"), "w") as log_file:
            log_file.write(f"Chat created: {time.ctime()}\n")

        self.chats[chat_id] = {
            "path": chat_path,
            "messages": [{"role": "system", "content": prompt}],
            "description": description
        }
        self.active_chat = chat_id

        return chat_id

    # TODO Fix this / make it acutally useful
    def _generate_description(self, initial_prompt):
        return f"Chat: {initial_prompt[:50]}{'...' if len(initial_prompt) > 50 else ''}"

    def get_chat_descriptions(self):
        descriptions = {}
        for chat_id, chat_data in self.chats.items():
            descriptions[chat_id] = chat_data["description"]
        return descriptions

    def list_chats(self):
        return list(self.chats.keys())

    def switch_chat(self, chat_id):
        if chat_id in self.chats:
            self.active_chat = chat_id
        else:
            raise ValueError("Invalid chat ID")

    def delete_chat(self, chat_id):
        if chat_id in self.chats:
            chat_path = self.chats[chat_id]["path"]
            shutil.rmtree(chat_path)
            del self.chats[chat_id]

            if self.active_chat == chat_id:
                self.active_chat = None
        else:
            raise ValueError("Invalid chat ID")

    def submit_prompt(self, chat_id, branch_id, prompt):
        chat_context = self.get_chat_context(chat_id, branch_id)
        context_prompt = f"{chat_context}\nUser: {prompt}\nAI:"
        response = self.generate_response(context_prompt)

        self.chats[chat_id]["branches"][branch_id]["messages"].append({"role": "user", "content": prompt})
        self.chats[chat_id]["branches"][branch_id]["messages"].append({"role": "ai", "content": response["text"]})

        self._log_response(chat_id, branch_id, response)

        return response

    def summarize_chat_block(self, chat_id, summary_level=1):
        messages = self.chats[chat_id]["messages"]
        
        # This is an absolute trash implementation ; wtf computer? come on. seriously. Fine. I'll do it myself..
        if summary_level == 1:
            recent_messages = messages[-(self.auto_summarize_interval * 2):]
        elif summary_level == 2:
            recent_messages = [msg for msg in messages if msg["summary_level"] == 1][-self.auto_higher_order_summarize_interval:]
        else:
            raise ValueError("Invalid summary level")

        summary_prompt = "Please provide a brief summary of the following conversation block:\n"
        for msg in recent_messages:
            summary_prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"

        summary_response = self.generate_response(summary_prompt)

        self.chats[chat_id]["messages"].append({"role": "summary", "content": summary_response["text"], "summary_level": summary_level})

        # TODO Extract to method
        # Save the chat thread summary data to a file
        chat_path = self.chats[chat_id]["path"]
        summary_file_path = os.path.join(chat_path, "summary.json")
        with open(summary_file_path, "w") as summary_file:
            json.dump(self.chats[chat_id], summary_file, indent=2)

    def get_summaries(self, chat_id, summary_level=1):
        return self.chats[chat_id]["summaries"][summary_level - 1]

    def print_summaries(self, chat_id, summary_level=1):
        summaries = self.get_summaries(chat_id, summary_level)
        for i, summary in enumerate(summaries):
            print(f"Summary {i + 1} (level {summary_level}): {summary['content']}")

   def get_chat_context(self, chat_id, branch_id):
        messages = self.chats[chat_id]["branches"][branch_id]["messages"]
        context = ""
        for message in messages:
            context += f"{message['role'].capitalize()}: {message['content']}\n"
        return context.strip()
    
    def _default_stream_callback(self, chunk_text):
        print(chunk_text, end="", flush=True)

    def set_stream_callback_to_file(self, file_path):
        def file_stream_callback(chunk_text):
            with open(file_path, "a") as f:
                f.write(chunk_text)

        self.stream_callback = file_stream_callback

    # TODO Switch this to `ChatCompletions` endpoint
    def generate_response(self, prompt, max_tokens=100, stream=False):
        if not stream:
            response = openai.Completion.create(
                model="gpt-4",
                prompt=prompt,
                max_tokens=max_tokens,
                n=1,
                stop=None,
                temperature=1
            )
            if self.active_chat is not None:
                self._log_response(response)

            return response.choices[0]
        else:
            response = openai.Completion.iter_chunks(
                model="gpt-4",
                prompt=prompt,
                max_tokens=max_tokens,
                n=1,
                stop=None,
                temperature=1
            )
            full_text = ""
            for chunk in response:
                full_text += chunk.text
                self.stream_callback(chunk.text)

            if self.active_chat is not None:
                self._log_response(response)

            return full_text

    def get_chat_summaries(self):
        summaries = {}
        for chat_id, chat_data in self.chats.items():
            duration = time.time() - float(chat_id)
            human_readable_duration = self._format_duration(duration)
            message_count = len(chat_data["messages"])
            context_length = sum(len(message["content"]) for message in chat_data["messages"])
            summaries[chat_id] = {
                "description": chat_data["description"],
                "duration": {
                    "raw": duration,
                    "human_readable": human_readable_duration
                },
                "message_count": message_count,
                "context_length": {
                    "value": context_length,
                    "units": "characters"
                }
            }
        return summaries

    def _format_duration(self, duration):
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

    def summarize_chat_progress(self, chat_id, num_prompts=5):
        if chat_id in self.chats:
            messages = self.chats[chat_id]["messages"]
            user_prompts = [m["content"] for m in messages if m["role"] == "user"][-num_prompts:]
            summary_prompt = "User's objectives in the last few messages: " + ' '.join(user_prompts) + "\nSummarize:"
            sub_chat_id = self.create_chat(initial_prompt="Summary chat", description="Summary of user's objectives")
            summary_response = self.submit_prompt(chat_id=sub_chat_id, prompt=summary_prompt)
            return summary_response
        else:
            raise ValueError("Invalid chat ID")

    def print_chat_summary_history(self, chat_id):
        if chat_id in self.chats:
            chat_data = self.chats[chat_id]
            summaries = [msg for msg in chat_data["messages"] if msg["role"] == "summary"]
            print(f"Summary history for chat {chat_id} ({chat_data['description']}):")
            for idx, summary in enumerate(summaries, 1):
                print(f"{idx}. {summary['content']}")
        else:
            raise ValueError("Invalid chat ID")


    # Branching functionality
    def create_branch(self, chat_id, initial_prompt=None):
        if chat_id not in self.chats:
            raise ValueError("Invalid chat ID")

        if initial_prompt is None:
            initial_prompt = "This is the start of a new branch."

        branch_id = str(time.time()).replace(".", "")

        branch_path = os.path.join(self.chats[chat_id]["path"], "branches", branch_id)
        os.makedirs(branch_path, exist_ok=True)

        with open(os.path.join(branch_path, "log.txt"), "w") as log_file:
            log_file.write(f"Branch created: {time.ctime()}\n")

        branch_data = {
            "path": branch_path,
            "parent_chat_id": chat_id,
            "messages": [{"role": "system", "content": initial_prompt}]
        }

        self.chats[chat_id]["branches"][branch_id] = branch_data

        return branch_id


    def list_branches(self, chat_id):
        return list(self.chats[chat_id]["branches"].keys())

    def switch_branch(self, branch_id):
        chat_id = branch_id.split("_")[0]
        if branch_id in self.chats[chat_id]["branches"]:
            self.active_chat = branch_id
        else:
            raise ValueError("Invalid branch ID")

    def create_checkpoint(self, chat_id, message_index):
        checkpoint_id = f"{chat_id}_checkpoint_{str(len(self.chats[chat_id]['checkpoints']) + 1)}"
        self.chats[chat_id]["checkpoints"][checkpoint_id] = message_index
        return checkpoint_id

    def list_checkpoints(self, chat_id):
        return list(self.chats[chat_id]["checkpoints"].keys())

    def submit_prompt_from_checkpoint(self, checkpoint_id, prompt):
        chat_id, branch_id, message_index = self._parse_checkpoint_id(checkpoint_id)
        new_branch_id = self.create_branch(chat_id)
        branch_data = self.chats[chat_id]["branches"][branch_id]

        # Copy messages up to the checkpoint to the new branch
        for message in branch_data["messages"][:message_index + 1]:
            self.chats[chat_id]["branches"][new_branch_id]["messages"].append(message)

        # Submit the new prompt to the new branch
        return self.submit_prompt(chat_id, new_branch_id, prompt)


    def _log_response(self, chat_id, branch_id, response):
        branch_data = self.chats[chat_id]["branches"][branch_id]
        prompt_id = response["id"]

        prompt_dir = os.path.join(branch_data["path"], prompt_id)
        os.makedirs(prompt_dir, exist_ok=True)

        index_data = {
            "prompt": response["prompt"],
            "prompt_id": prompt_id,
            "received_time": time.time(),
            "response_time": response["created"],
            "response_duration": response["usage"]["total_tokens"],
            "response_choices": [choice["text"] for choice in response.choices]
        }

        with open(os.path.join(prompt_dir, "index.json"), "w") as index_file:
            json.dump(index_data, index_file, indent=2)

        with open(os.path.join(branch_data["path"], "log.txt"), "a") as log_file:
            log_file.write(f"User: {response['prompt']}\n")
            log_file.write(f"AI: {response.choices[0]['text']}\n")
            log_file.write("\n")

    # WIP - Implement different 'modes' so to e.g. overwrite changes in docs vs append-only, etc

    def set_project_type(self, project_type):
        self.working_on_project = project_type

