import logging
from multiprocessing import JoinableQueue, Process
from time import sleep
from typing import List, Any, Dict

from flask_socketio import SocketIO

from mongoDAO.MongoDAO import MongoDAO
from mongoDAO.chatAIDescriptionRepository import clear_chatai_descriptions, find_all_chatai_descriptions, \
    add_chatai_description
from mongoDAO.studentActionRepository import update_chat_ai_answer, set_chat_ai_achieved
from mongoModel.ChatAIDescription import ChatAIDescription
from mongoModel.Exam import Exam
from mongoModel.SocratQuestionnaire import SocratQuestionnaire
from mongoModel.StudentAction import AskChatAI
from services.chatAI.ChatAIHandler import ChatAIHandler
from services.chatAI.CopyPasteHandler import CopyPasteHandler
from services.chatAI.DalaiHandler import DalaiHandler
from services.chatAI.OpenAIHandler import OpenAIHAndler
from utils.Singleton import Singleton

__all__ = ['ChatAIManager']

from utils.loggingUtils import configure_logging

LOG = logging.getLogger(__name__)


def handle_chat_answer(config: Dict, queue: JoinableQueue):
    # Since this function will be a new process, we need to re-configure logging, and ask for new connection
    # (socketio, mongo)
    configure_logging(config.get('LOG_LEVEL'))
    try:
        redis_url = config.get('REDIS_URL', 'redis://')
        socketio = SocketIO(message_queue=redis_url)
        with MongoDAO(configuration=MongoDAO.compute_dao_options_from_app(config)) as mongo_dao:
            while True:
                response: Dict = queue.get()
                queue.task_done()

                request_type = response.get('request_type')
                if request_type == 'model':
                    chat_key = response.get('chat_key')
                    if chat_key is None:
                        LOG.warning('Receive model answer with chat key')
                        continue
                    model_key = response.get('answer')
                    if model_key is not None:
                        description = ChatAIDescription(chat_key=chat_key, model_key=model_key)
                        add_chatai_description(mongo_dao, description)
                        LOG.info("New Chat AI Model discovered! %s : %s", chat_key, model_key)

                elif request_type == 'prompt':
                    action_id = response.get('action_id')
                    user_sid = response.get('user_sid')
                    if action_id is None or user_sid is None:
                        LOG.warning('Receive response without chat_key or model_key or action_id or user_sid')
                        continue
                    # if answer is present, update action in db
                    answer = response.get('answer')
                    achieved = response.get('ended', False)
                    if answer is not None:
                        update_chat_ai_answer(mongo_dao, action_id, answer, achieved=achieved)
                    elif achieved is True:
                        set_chat_ai_achieved(mongo_dao, action_id, achieved=achieved)
                    # remove user_sid from response to relay it
                    response.pop('user_sid')
                    socketio.emit('answer', response, room=user_sid)
                else:
                    LOG.warning('Receive response without allowed request_type: %s.', request_type)
    except KeyboardInterrupt:
        print("Stop AI answer handling process")
    finally:
        queue.close()


class ChatAIManager(metaclass=Singleton):
    def __init__(self, config: Dict = None):
        self._answer_queue: JoinableQueue = JoinableQueue()
        self._answer_process: Process = None
        self._ai_handlers_by_chat_key: Dict[str, ChatAIHandler] = {}
        self._config: Dict = config
        self._configure_handlers()

    def _configure_handlers(self):
        if self._config is not None:
            # According to config, instanciate different handler
            if 'CHATAI_DALAI_URL' in self._config:
                h = DalaiHandler(self._answer_queue, self._config)
                self._ai_handlers_by_chat_key[h.chat_key] = h
            if self._config.get('CHATAI_OPENAI_ENABLED', False) is True:
                h = OpenAIHAndler(self._answer_queue, self._config)
                self._ai_handlers_by_chat_key[h.chat_key] = h
        # Add copyPast
        h = CopyPasteHandler(self._answer_queue)
        self._ai_handlers_by_chat_key[h.chat_key] = h

    def start(self):
        self._answer_process = Process(target=handle_chat_answer, args=(self._config, self._answer_queue))
        self._answer_process.start()
        # Create connect all handler
        LOG.info("connect to all AI handler")
        for handler in self._ai_handlers_by_chat_key.values():
            LOG.info("Connection to handler %s...", handler.name)
            handler.connect()
        # Wait for all handler to be connected
        LOG.info("Wait for all handler to be connected")
        max_wait = 5
        current_wait = 0
        while current_wait < max_wait \
                and any(filter(lambda h: not h.connected, self._ai_handlers_by_chat_key.values())):
            current_wait += 1
            LOG.debug("Wait 1 sec for chat ai handler to be connected")
            sleep(1)
        connected_handlers = list(filter(lambda h: h.connected, self._ai_handlers_by_chat_key.values()))
        if len(connected_handlers) < len(self._ai_handlers_by_chat_key):
            LOG.warning("Warning: not all chat AI handler connected")
            for key, handler in self._ai_handlers_by_chat_key.items():
                LOG.debug("- {}: {}".format(key, handler.connected))
        # clear model desc in db
        LOG.debug("Clear chat AI model description in database")
        mongo_dao = MongoDAO()
        clear_chatai_descriptions(mongo_dao)

        # for all connected handler, ask their model
        LOG.debug("Ask handler their models")
        request_identifiers = dict(request_type='model')
        for handler in connected_handlers:
            handler.request_available_models(request_identifiers=request_identifiers)
        LOG.info("Chat AI Service ready")

    def _generate_available_chats(self) -> Dict:
        mongo_dao = MongoDAO()
        for chat_desc in find_all_chatai_descriptions(mongo_dao):
            chat_key = chat_desc['chat_key']
            model_key = chat_desc['model_key']
            handler = self._ai_handlers_by_chat_key.get(chat_key)
            if handler is None:
                LOG.warning("Got model associated to a chat key without any handler")
                continue
            model_title = handler.get_model_name(model_key)
            if model_title is None:
                model_title = 'Unknown/No model'
            yield {
                'id': "{}.{}".format(chat_key, model_key),
                'chat_key': chat_key,
                'model_key': model_key,
                'title': "{}. {}.".format(handler.name, model_title),
                'copyPaste': handler.copy_past,
                'privateKeyRequired': handler.private_key_required
            }

    @property
    def available_chats(self) -> List[Dict]:
        return list(self._generate_available_chats())

    def compute_student_choice_from_exam(self, exam: Exam) -> Any:
        available_chat_dict = dict((c['id'], c) for c in self.available_chats)
        choices = []
        for id in exam['selected_chats'].keys():
            chat_info = available_chat_dict.get(id)
            if chat_info is None:
                continue
            choice = dict(id=id, chat_key=chat_info['chat_key'], model_key=chat_info['model_key'],
                          title=chat_info['title'])
            if chat_info.get('copyPaste') is True:
                choice['copyPaste'] = True
            choices.append(choice)
        return choices

    def compute_student_choice_from_socrat(self, socrat: SocratQuestionnaire) -> Any:
        available_chat_dict = dict((c['id'], c) for c in self.available_chats)
        choices = []
        id = socrat['selected_chat'].get('id')
        if id:
            chat_info = available_chat_dict.get(id)
            if chat_info:
                choice = dict(id=id, chat_key=chat_info['chat_key'], model_key=chat_info['model_key'],
                              title=chat_info['title'])
                if chat_info.get('copyPaste') is True:
                    choice['copyPaste'] = True
                choices.append(choice)
        return choices

    def process_prompt(self, action_id: str, action: AskChatAI, user_sid: str,
                       private_key: str = None, custom_init_prompt: str = None, **kwargs) -> None:
        prompt = action['prompt'] if action.get('prompt') else action.get('hidden_prompt')
        if prompt is None:
            raise Exception("No prompt to process")
        handler = self._ai_handlers_by_chat_key.get(action['chat_key'])
        if not handler:
            raise Exception("Unmanaged chat: {}".format(action['chat_key']))
        request_identifiers = dict(request_type='prompt', question_idx=action['question_idx'],
                                   action_id=action_id, user_sid=user_sid, chat_id=action['chat_id'])
        handler.send_prompt(action['model_key'], prompt, request_identifiers, private_key=private_key,
                            action=action, custom_init_prompt=custom_init_prompt, **kwargs)
