from io import BytesIO, BufferedReader
from openai import OpenAI
from json import loads
from re import sub
from enum import Enum

def create_client(api_key=str) -> OpenAI:
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        raise Exception(f'Cannot Configure Client, Invalid API Key: {e}')


class GPTRole(Enum):
    USER = 'user'
    SYSTEM = 'system'
    ASSISTANT = 'assistant'


class GPTChatCompletion:
    def __init__(self, client: OpenAI, language: str, isResponseJson: bool = False) -> None:
        self.messages = []
        self.client = client
        self.isResponseJson = isResponseJson
        self.add_message(GPTRole.SYSTEM, f'Please return the response in {language}')

    def add_message(self, role: GPTRole, msg: str):
        self.messages.append({'role': role.value, 'content': msg})

    def get_completion(self):
        response_type = 'json_object' if self.isResponseJson else 'text'
        response = self.client.chat.completions.create(
            model='gpt-3.5-turbo-1106',
            response_format={'type': response_type},
            messages=self.messages)
        return response.choices[0].message.content


class GPTAssistant:
    def __init__(self, client: OpenAI) -> None:
        self.client = client
        self.assistant, self.thread = None, None
        self.files = []
        self.language = 'Spanish'
        self.hasBuilt = False

    def config_language(self, language: str):
        self.language = language

    def add_file(self, file_id: str):
        self.files.append(file_id)

    def upload_file(self, iep: BytesIO) -> str:
        file = self.client.files.create(
            file=BufferedReader(iep), purpose='assistants')
        return file.id

    def build(self, instructions: str) -> str:
        print(self.files)
        assistant = self.client.beta.assistants.create(
            name='IEP Chatbot',
            instructions=f'{instructions}. Please return the response in {self.language}',
            tools=[{'type': 'retrieval'}],
            model='gpt-4-1106-preview',
            file_ids=self.files)
        print('A')
        self.assistant_id = assistant.id  # Need Validation
        self.hasBuilt = True
        return assistant.id

    def add_message(self, message: str) -> str:
        thread = self.client.beta.threads.create(
            messages=[{'role': 'user', 'content': message}])
        self.thread_id = thread.id  # Need Validation
        return thread.id

    def _check_configuration(self):
        if not self.assistant_id or not self.thread_id:
            raise Exception('IEP and/or Messages Not Configured')

    def run(self) -> str:
        self._check_configuration()
        run = self.client.beta.threads.runs.create(
            thread_id=self.thread_id,
            assistant_id=self.assistant_id,
            instructions=f'Please return the response in {self.language}')
        self.run_id = run.id  # Need Validation
        return run.id

    def get_status(self) -> tuple:
        if not self.run_id:
            raise Exception('Assistant Not Started Yet')
        run = self.client.beta.threads.runs.retrieve(
            thread_id=self.thread_id,
            run_id=self.run_id)
        return run.created_at, run.completed_at, run.expires_at, run.cancelled_at

    def has_finished(self) -> bool:
        return self.get_status()[1]

    def get_latest_message(self) -> str:
        if not self.has_finished():
            raise Exception('Assistant Not Finished Yet')
        messages = self.client.beta.threads.messages.list(
            thread_id=self.thread_id)
        messages_data = loads(messages.model_dump_json())
        print(messages_data)
        latest_response = messages_data['data'][0]['content'][0]['text']['value']
        return sub(r'\【.*?\】', '', latest_response)
