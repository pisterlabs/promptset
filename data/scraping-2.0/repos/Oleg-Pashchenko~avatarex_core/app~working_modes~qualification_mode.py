import dataclasses
import json

from openai import OpenAI

from app.sources.amocrm import methods, new_amo
from app.sources.amocrm.db import PipelineSettings, AvatarexSiteMethods, AmocrmSettings
from app.sources.amocrm.new_amo import AmoConnect
from app.utils.db import MethodResponse, Command, Message
from app.working_modes.knowledge_mode import perephrase


def get_field_name_by_question(q, ff):
    for f in ff.keys():
        if ff[f] == q:
            return f


@dataclasses.dataclass
class QualificationMode:
    @staticmethod
    def _get_qualification_question(field_number, source_fields, fields_to_fill):
        count = 0
        for field in fields_to_fill.keys():
            if source_fields[field]['active'] is None:
                count += 1
                if count == field_number:
                    return fields_to_fill[field], source_fields[field]
        return None, None

    @staticmethod
    def is_this_answer_for_this_question(question, answer, openai_key, field):
        if field['type'] != 'field':
            required = ['param']
            properties = {'param': {'type': 'string', 'enum': []}}
            for f in field['values']:
                properties['param']['enum'].append(f['value'])

        else:
            required = ['is_correct']
            properties = {'is_correct':
                              {'type': 'boolean',
                               'description': question,
                               }}
        func = [{
            "name": "Function",
            "description": "Function description",
            "parameters": {
                "type": "object",
                "properties": properties,
                'required': required
            }
        }]

        try:
            messages = [
                {'role': 'system', 'content': 'Give answer:'},
                {"role": "user",
                 "content": answer}]

            client = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(model="gpt-3.5-turbo-0613",
                                                      messages=messages,
                                                      functions=func,
                                                      function_call="auto")
            response_message = response.choices[0].message
        except:
            return False
        if response_message.function_call:
            function_args = json.loads(response_message.function_call.arguments)
            if 'is_correct' in function_args:
                if function_args['is_correct'] is True:
                    return True, answer
                return False, ''
            else:
                return True, function_args['param']
        else:
            return False, ''

    @staticmethod
    def _check_user_answer(fields_to_fill, source_fields, message, openai_key) -> (bool, Command | None):
        question, field = QualificationMode._get_qualification_question(1, source_fields, fields_to_fill)
        is_correct, v = QualificationMode.is_this_answer_for_this_question(question, message, openai_key, field)
        if is_correct:
            return True, Command("fill", {
                'question': question,
                'value': v,
                'name': source_fields[get_field_name_by_question(question, fields_to_fill)]})
        return False, None

    @staticmethod
    def _is_qualification_passed(fields_to_fill: dict, source_fields: dict) -> bool:  # if > 0: get question
        for field_to_fill in fields_to_fill.keys():
            if source_fields[field_to_fill]['active'] is None:  # Если поле не создано
                return False
        return True

    def execute(self, fields_to_fill: dict, amocrm_settings: AmocrmSettings, lead_id: int, message, openai_key,
                q_f_message) -> (
            MethodResponse, bool, bool):
        amo_connection = AmoConnect(amocrm_settings.mail, amocrm_settings.password, host=amocrm_settings.host, deal_id=lead_id)
        amo_connection.auth()
        source_fields = amo_connection.get_params_information(list(fields_to_fill.keys()))

        # source_fields = methods.get_fields_info(amocrm_settings, lead_id, fields_to_fill)

        if len(fields_to_fill.keys()) == 0:  # если пользователь выставил что ничего заполнять не нужно
            return MethodResponse(data=[], all_is_ok=True, errors=set()), None, False

        if self._is_qualification_passed(fields_to_fill, source_fields):  # если квалификация уже пройдена
            return MethodResponse(data=[], all_is_ok=True, errors=set()), None, False
        # return MethodResponse(data=[], all_is_ok=True, errors=set()), True, True
        # если все же мы остались здесь, значит нужно проверить ответ и задать квалифициирующий вопрос

        is_answer_correct, command = self._check_user_answer(fields_to_fill, source_fields, message, openai_key)
        data = []
        if is_answer_correct:  # если ответ принят
            data.append(command)  # добавляем команду на заполнение поля
            message, _ = self._get_qualification_question(2,
                                                           source_fields, fields_to_fill)  # просим следующее сообщение
        else:
            message, _ = self._get_qualification_question(1, source_fields, fields_to_fill)  # повторяем текущий вопрос

        if message:  # если. сообщение сформировалось
            # perephrase message
            data.append(Message(perephrase(api_key=openai_key, message=message)))

        if is_answer_correct is True and message is None:
            data.append(Message(perephrase(api_key=openai_key, message=q_f_message)))

        return MethodResponse(all_is_ok=True, errors=set(), data=data), is_answer_correct, message is not None

    @staticmethod
    def execute_amocrm(pipeline_settings: PipelineSettings, amocrm_settings: AmocrmSettings,
                       lead_id: int, message, openai_key) -> (MethodResponse, bool, bool):
        print('Запущена квалификация')
        # временный костыль для AmoCRM
        if pipeline_settings.chosen_work_mode == 'Ответ по контексту' or pipeline_settings.chosen_work_mode == 'Prompt mode':
            data = AvatarexSiteMethods.get_prompt_method_data(pipeline_settings.p_mode_id)
            return QualificationMode().execute(data.qualification, amocrm_settings, lead_id, message, openai_key,
                                               data.qualification_finished)

        elif pipeline_settings.chosen_work_mode == 'Ответ из базы знаний':
            print('for knowledge mode')
            data = AvatarexSiteMethods.get_knowledge_method_data(pipeline_settings.k_mode_id)
            print(data)
            return QualificationMode().execute(data.qualification, amocrm_settings, lead_id, message, openai_key,
                                               data.qualification_finished)

        elif pipeline_settings.chosen_work_mode == 'Ответ из базы данных':
            print('for search mode')
            return MethodResponse(all_is_ok=True, data=[], errors=set()), None, False

        else:
            print('for knowledge and search mode')
            return MethodResponse(all_is_ok=True, data=[], errors=set()), None, False


"""
@dataclasses.dataclass
class QualificationModeData:
    fields_to_fill: list[dict]
    d_m_data: DatabaseModeData


class QualificationMode:
    # Мод который вызывается в случае если is_qualification_passed - False

    def __init__(self, data: KnowledgeAndSearchData, responses: list, errors: set):
        self.data: KnowledgeAndSearchData = data
        self.responses: list[Message | Command] = responses
        self.errors = errors

    async def is_qualification_passed(self) -> bool:  # if > 0: get question
        fields_to_fill = self.data.fields_to_fill
        for field in fields_to_fill:
            if not field['exists']:
                return False
        return True

    async def _is_qualification_question_answer_satisfy(self) -> bool:
        pass

    async def _get_unfiled_field(self, position_index: int) -> (dict, None):
        pass

    async def _create_question(self, unfiled_field: dict) -> bool:
        answer, status = _perephrase(message=unfiled_field['question'], data=self.data)

        if status:
            self.responses.append(Message(text=answer, set_last_order=True))
        else:
            self.responses.append(Message(text=unfiled_field['question'], set_last_order=True))
            self.errors.add(err.OPENAI_REQUEST_ERROR)
        return status

    async def execute(self):
        unfiled_field = await self._get_unfiled_field(1)  # получаем первое незаполненное поле
        if unfiled_field is None:  # если все поля заполнены
            return SearchMode(data=self.data, responses=self.responses, errors=self.errors).execute()

        if self.is_message_first:  # если это первое сообщение от клиента
            status = await self._create_question(unfiled_field=unfiled_field)
            return MethodResponse(data=self.responses, all_is_ok=status, errors=self.errors)

        unfiled_field = await self._get_unfiled_field(2)  # получаем второе незаполненное поле
        if unfiled_field is None:  # если все поля заполнены
            return SearchMode(data=self.data, responses=self.responses, errors=self.errors,
                              ).execute()

        else:
            if not await self._is_qualification_question_answer_satisfy():
                # если мы посчитали что ответ на вопрос не был дан или был дан некорректно
                return SearchMode(data=self.data, responses=self.responses, errors=self.errors,
                                  ).execute()
            else:
                # если ответ нас устроил
                self.responses.append(Command(command='fill', data=unfiled_field))
                status = await self._create_question(unfiled_field=unfiled_field)  # Создаем новый вопрос
                return MethodResponse(data=self.responses, all_is_ok=status, errors=self.errors)
"""
