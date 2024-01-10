import logging
import json
import zlib
import uuid
import time
from pathlib import Path
from .model import model
from . import prompt
from . import config
from . import utils
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from cryptography.fernet import Fernet, InvalidToken
logger = logging.getLogger('heymans')


class Messages:
    
    def __init__(self, heymans, persistent=False):
        self._heymans = heymans
        self._persistent = persistent
        self._session_folder = Path(config.sessions_folder)
        if not self._session_folder.exists():
            self._session_folder.mkdir()
        self._session_path = self._session_folder / self._heymans.user_id
        self._conversation_id = str(uuid.uuid4())
        self._conversation_title = config.default_conversation_title
        self._message_history = []
        self._message_metadata = []
        self._condensed_text = None
        self._session = {
            'conversations': {
                self._conversation_id: {
                    'title': self._conversation_title,
                    'condensed_text': self._condensed_text,
                    'message_history': self._message_history,
                    'message_metadata': self._message_metadata,
                    'last_updated': time.time()
                }
            },
            'active_conversation': self._conversation_id}
        self.clear()
        if self._persistent:
            self._fernet = Fernet(self._heymans.encryption_key)
            self.load()
        
    def __len__(self):
        return len(self._message_history)
        
    def __iter__(self):
        for msg in self._message_history:
            yield msg
            
    def list_conversations(self):
        return {conversation_id: [conversation['title'],
                                  conversation['last_updated']]
                for conversation_id, conversation
                in self._session['conversations'].items()}
    
    def new_conversation(self):
        self._conversation_id = str(uuid.uuid4())
        self._conversation_title = config.default_conversation_title
        self._message_history = []
        self._message_metadata = []
        self._condensed_text = None
        self._session['active_conversation'] = self._conversation_id
        logger.info(f'new conversation {self._conversation_id}')
        self.clear()

    def activate_conversation(self, conversation_id):
        if conversation_id not in self._session['conversations']:
            logger.info(
                f'cannot activate non-existing conversation {conversation_id}')
            return
        self._load_conversation(conversation_id)
        
    def delete_conversation(self, conversation_id):
        if conversation_id not in self._session['conversations']:
            logger.info(
                f'cannot delete non-existing conversation {conversation_id}')
            return
        del self._session['conversations'][conversation_id]
        
    def _load_conversation(self, conversation_id):
        self._conversation_id = conversation_id
        logger.info(f'loading conversation {self._conversation_id}')
        conversation = self._session['conversations'].get(
            self._conversation_id, {})
        self._condensed_text = conversation.get('condensed_text', None)
        self._message_history = conversation.get('message_history', [])
        self._conversation_title = conversation.get(
            'title', 'Untitled conversation')
        self._condensed_message_history = conversation.get(
            'condensed_message_history', [])
        self._message_metadata = conversation.get('message_metadata', [])
        self._session['active_conversation'] = conversation_id
            
    def clear(self):
        self._condensed_text = None
        metadata = self.metadata()
        metadata['search_model'] = 'Welcome message'
        self._message_history = [('assistant', self.welcome_message(), 
                                  metadata)]
        self._condensed_message_history = [
            (role, content) for role, content, metadata
            in self._message_history[:]]        
            
    def metadata(self):
        return {'timestamp': utils.current_datetime(),
                'sources': self._heymans.documentation.to_json(),
                'search_model': config.search_model,
                'condense_model': config.condense_model,
                'answer_model': config.answer_model}
        
    def append(self, role, message):
        metadata = self.metadata()
        self._message_history.append((role, message, metadata))
        self._condensed_message_history.append((role, message))
        self._condense_message_history()
        if self._persistent:
            self.save()
        return metadata
    
    def prompt(self):
        prompt = [SystemMessage(content=self._system_prompt())]
        for role, content in self._condensed_message_history:
            if role == 'assistant':
                prompt.append(AIMessage(content=content))
            elif role == 'user':
                prompt.append(HumanMessage(content=content))
            else:
                raise ValueError(f'Invalid role: {role}')
        return prompt

    def welcome_message(self):
        return config.welcome_message
        
    def _condense_message_history(self):
        system_prompt = self._system_prompt()
        messages = [{"role": "system", "content": system_prompt}]
        prompt_length = sum(len(content) for role, content
                            in self._condensed_message_history)
        logger.info(f'system prompt length: {len(system_prompt)}')
        logger.info(f'prompt length (without system prompt): {prompt_length}')
        if prompt_length <= config.max_prompt_length:
            logger.info('no need to condense')
            return
        condense_length = 0
        condense_messages = []
        while condense_length < config.condense_chunk_length:
            role, content = self._condensed_message_history.pop(0)
            condense_length += len(content)
            condense_messages.insert(0, (role, content))
        logger.info(f'condensing {len(condense_messages)} messages')
        condense_prompt = prompt.render(
            prompt.CONDENSE_HISTORY,
            history=''.join(
                f'You said: {content}' if role == 'system' else f'User said: {content}'
                for role, content in condense_messages))
        self._condensed_text = self._heymans.condense_model.predict(
            condense_prompt)
        
    def _system_prompt(self):
        system_prompt = [prompt.render(
            self._heymans.system_prompt,
            current_datetime=utils.current_datetime())]
        for tool in self._heymans.tools.values():
            if tool.prompt:
                system_prompt.append(tool.prompt)
        if len(self._heymans.documentation):
            system_prompt.append(self._heymans.documentation.prompt())
        if self._condensed_text:
            logger.info('appending condensed text to system prompt')
            system_prompt.append(prompt.render(
                prompt.SYSTEM_PROMPT_CONDENSED,
                summary=self._condensed_text))
        return '\n\n'.join(system_prompt)
        
    def _update_title(self):
        """The conversation title is updated when there are at least two 
        messages, excluding the system prompt and AI welcome message. Based on
        the last messages, a summary title is then created.
        """
        if len(self) <= 2 or \
                self._conversation_title != config.default_conversation_title:
            return
        title_prompt = [SystemMessage(content=prompt.TITLE_PROMPT)]
        title_prompt += self.prompt()[2:]
        self._conversation_title = self._heymans.condense_model.predict(
            title_prompt).strip('"\'')
        if len(self._conversation_title) > 100:
            self._conversation_title = self._conversation_title[:100] + '…'
        print(f'new conversation title: {self._conversation_title}')

    def load(self):
        if not self._session_path.exists():
            logger.info(f'starting new session: {self._session_path}')
            return
        logger.info(f'loading session file: {self._session_path}')
        try:
            self._session = json.loads(self._fernet.decrypt(
                self._session_path.read_bytes()).decode('utf-8'))
        except (json.JSONDecodeError, InvalidToken) as e:
            logger.warning(f'failed to load session file: {self._session_path}: {e}')
            return
        if not isinstance(self._session, dict):
            logger.warning(f'session file invalid: {self._session_path}')
            return
        if 'active_conversation' not in self._session:
            logger.warning('no active conversation to load')
            return
        self._load_conversation(self._session['active_conversation'])
    
    def save(self):
        self._update_title()
        conversation = {
            'condensed_text': self._condensed_text,
            'message_history': self._message_history,
            'condensed_message_history': self._condensed_message_history,
            'title': self._conversation_title,
            'last_updated': time.time()
        }
        self._session['conversations'][self._conversation_id] = conversation
        logger.info(f'saving session file: {self._session_path}')
        self._session_path.write_bytes(
            self._fernet.encrypt(json.dumps(self._session).encode('utf-8')))
