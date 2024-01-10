from typing import Type

from openai.types.audio import Translation
from pydantic import BaseModel, HttpUrl, ValidationError

from autobots.action.action_type.abc.IAction import IAction, ActionConfigType, ActionInputType, ActionOutputType
from autobots.action.action_type.action_types import ActionType
from autobots.conn.openai.openai_client import get_openai
from autobots.conn.openai.openai_audio.translation_model import TranslationReq
from autobots.core.logging.log import Log


class AudioRes(BaseModel):
    url: str


class ActionAudio2TextTranslationOpenai(IAction[TranslationReq, AudioRes, Translation]):
    type = ActionType.audio2text_translation_openai

    @staticmethod
    def get_config_type() -> Type[ActionConfigType]:
        return TranslationReq

    @staticmethod
    def get_input_type() -> Type[ActionInputType]:
        return AudioRes

    @staticmethod
    def get_output_type() -> Type[ActionOutputType]:
        return Translation

    def __init__(self, action_config: TranslationReq):
        super().__init__(action_config)

    async def run_action(self, action_input: AudioRes) -> Translation | None:
        try:
            if self.action_config.file_url is None and action_input.url is None:
                return None
            if action_input.url is not None:
                self.action_config.file_url = HttpUrl(action_input.url)

            translation = await get_openai().openai_audio.translation(self.action_config)
            return translation
        except ValidationError as e:
            Log.error(str(e))
        except Exception as e:
            Log.error(str(e))
