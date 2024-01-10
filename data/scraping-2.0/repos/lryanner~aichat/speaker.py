import os
import random
import re
import string
import time

import numpy as np
import openai
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

import utils
from AIChatEnum import SpeakerAPIType
from gradio_client import Client

from data import VITSConfigData, VITSConfigDataList
from exceptions import SpeakerException


class Speaker:
    def __init__(self, vits_config: VITSConfigDataList,
                 emotion_mapping_path='./resources/mapping/emotion_no_duplicated.csv',
                 marked_emotion_mapping_path='./resources/mapping/nene_emotion_mapping.json',
                 dialogue_emotion_ordering_mapping_path='./resources/mapping/dialogue_emotion_ordering_mapping.json',
                 dialogues_emotion_mapping_path='./resources/mapping/dialogues_emotion_mapping.json',
                 dialogues_emotion_mapping_npy_path='./resources/mapping/dialogues_emotion_mapping.npy', ):
        """
        The speaker class.
        :param vits_config: The vits config.
        :param emotion_mapping_path: [optional] The path of the nene emotion mapping.
        """
        self._config = vits_config
        self._emotion_mapping_path = emotion_mapping_path
        self._marked_emotion_mapping_path = marked_emotion_mapping_path
        self._dialogue_emotion_ordering_mapping_path = dialogue_emotion_ordering_mapping_path
        self._dialogues_emotion_mapping_path = dialogues_emotion_mapping_path
        self._dialogues_emotion_mapping_npy_path = dialogues_emotion_mapping_npy_path
        self._speaker = None

    def setup_config(self):
        """
        Set up the config.
        :return:
        """
        active_config = self._config.get_active_vits_config()
        emotion_mapping_path = self._emotion_mapping_path

        if active_config.api_type == SpeakerAPIType.NeneEmotion.value:
            self._speaker = SpeakerNeneEmotion(
                self.join_address(active_config.api_address, active_config.api_port),
                emotion_mapping_path,
                self._marked_emotion_mapping_path,
                self._dialogue_emotion_ordering_mapping_path,
                self._dialogues_emotion_mapping_path,
                self._dialogues_emotion_mapping_npy_path)
        elif active_config.api_type == SpeakerAPIType.VitsSimpleAPI.value:
            self._speaker = SpeakerVitsSimpleApi(
                self.join_address(active_config.api_address, active_config.api_port),
                emotion_mapping_path,
                self._marked_emotion_mapping_path,
                self._dialogue_emotion_ordering_mapping_path,
                self._dialogues_emotion_mapping_path,
                self._dialogues_emotion_mapping_npy_path)

    @staticmethod
    def join_address(api_address, api_port):
        """
        Join the api address and port.
        :param api_address: The api address.
        :param api_port: The api port.
        :return: The joined api address and port.
        """
        return 'http://' + api_address + ':' + str(api_port)

    def speak(self, text, **kwargs):
        """
        Speak the text.
        :param text:
        :param kwargs:
        :return:
        """
        return self._speaker(text, **kwargs)

    def play_emotion_sample_file(self, emotion_id, root_path):
        """
        Play the emotion sample file.
        :param emotion_id: the emotion id
        :param root_path: the root path of the emotion sample file
        :return: none
        """
        self._speaker.play_emotion_sample_file(emotion_id, root_path)

    def last_emotion_sample(self):
        """
        Get the emotion sample.
        :return: the emotion sample
        """
        return self._speaker.last_emotion_sample


class SpeakerW2V2:
    def __init__(self, api_address, emotion_mapping_path,
                 marked_emotion_mapping_path,
                 dialogue_emotion_ordering_mapping_path,
                 dialogues_emotion_mapping_path,
                 dialogues_emotion_mapping_npy_path):
        """
        The speaker class for nene emotion. This class is callable.

        :param api_address: Api address of nene emotion server.
        :param emotion_model_path: The path of the nene emotion mapping.
        """
        self._text_weight = 0.8
        self._context_weight = 1 - self._text_weight
        self._last_emotion_sample = None
        self._out_put_path = os.path.join(os.path.dirname(__file__), 'download\\sounds')
        self._api_address = api_address
        self._emotion_mapping = utils.load_csv(emotion_mapping_path)
        self._marked_emotion_mapping: dict[str:str] = utils.load_json(marked_emotion_mapping_path)
        self._all_emotions = [[emotion['arousal'], emotion['dominance'], emotion['valence']] for emotion in
                              self._emotion_mapping]
        self._nsfw_emotions = [[emotion['arousal'], emotion['dominance'], emotion['valence']] for emotion in
                               self._emotion_mapping if emotion['nsfw'] == 1]
        self._sfw_emotions = [[emotion['arousal'], emotion['dominance'], emotion['valence']] for emotion in
                              self._emotion_mapping if emotion['nsfw'] == 0]
        self._dialogue_emotion_ordering_mapping = utils.load_json(dialogue_emotion_ordering_mapping_path)
        self._dialogues_emotion_mapping = utils.load_json(dialogues_emotion_mapping_path)
        self._dialogues_emotion_mapping_npy = np.load(dialogues_emotion_mapping_npy_path)
        self._processed_dialogues_emotion_mapping_npy = self._dialogues_emotion_mapping_npy[:, 0,
                                                        :] * self._text_weight + self._dialogues_emotion_mapping_npy[:,
                                                                                 1, :] * self._context_weight

    @property
    def last_emotion_sample(self):
        return self._last_emotion_sample

    def _get_emotion_sample(self, emotion, nsfw=None):
        """
        Get the emotion sample.
        :param emotion: the emotion. Must be a list of float.
        :return:
        """
        if isinstance(emotion[0], float):
            if nsfw is None:
                self._last_emotion_sample = \
                    self._emotion_mapping[utils.get_similar_array_index(emotion, self._all_emotions)]['emotion']
            elif nsfw:
                self._last_emotion_sample = self._emotion_mapping[
                    utils.get_similar_array_index(utils.get_similar_array(emotion, self._nsfw_emotions),
                                                  self._all_emotions)]['emotion']
            else:
                self._last_emotion_sample = self._emotion_mapping[
                    utils.get_similar_array_index(utils.get_similar_array(emotion, self._sfw_emotions),
                                                  self._all_emotions)]['emotion']
            return self._last_emotion_sample
        elif isinstance(emotion[0], str):
            if nsfw:
                mapping = self._marked_emotion_mapping['nsfw']
            else:
                mapping = self._marked_emotion_mapping['safe']
            result = []
            if emotion[0] == '娇喘':
                # move to the end
                emotion = emotion[1:] + emotion[:1]
            for emo in emotion:
                if emo in mapping.keys():
                    if not result:
                        result = mapping[emo]
                    else:
                        same = utils.get_same_item(result, mapping[emo])
                        if same:
                            result = same
            return utils.shuffle_list(result)[0]

    def get_emotion_sample_by_text(self, text, context=None, translated_text=None):
        """
        Get the emotion sample by text.
        :param text: the text.
        :return:
        """
        if context:
            r = openai.Embedding.create(
                model='text-embedding-ada-002',
                input=[text, context]
            )
            text_embedding = np.array(r['data'][0]['embedding'])
            context_embedding = np.array(r['data'][1]['embedding'])
            result_embedding = text_embedding * self._text_weight + context_embedding * self._context_weight
        else:
            r = openai.Embedding.create(
                model='text-embedding-ada-002',
                input=text,
            )
            text_embedding = np.array(r['data'][0]['embedding'])
            result_embedding = text_embedding
        topn_closest = utils.find_topn_closest_indices(result_embedding, self._processed_dialogues_emotion_mapping_npy,
                                                       6).tolist()
        topn_closest_string = [list(self._dialogues_emotion_mapping.values())[index][0] for index in topn_closest]
        index = utils.find_closest_string(translated_text, topn_closest_string)
        return self._dialogue_emotion_ordering_mapping[str(topn_closest[index])]

    def play_emotion_sample_file(self, emotion, root):
        """
        Play the emotion sample file.
        :param emotion: the emotion. Must be a list of float.
        :param root: the root directory of the emotion sample.
        :return:
        """
        for data in self._emotion_mapping:
            if data['emotion'] == emotion:
                file_path = os.path.join(root, data['file'] + '.wav')
                utils.play_sound(file_path)
                return


class SpeakerNeneEmotion(SpeakerW2V2):
    def __init__(self, api_address,
                 emotion_mapping_path,
                 marked_emotion_mapping_path,
                 dialogue_emotion_ordering_mapping_path,
                 dialogues_emotion_mapping_path,
                 dialogues_emotion_mapping_npy_path):
        """
        The speaker class for nene emotion. This class is callable.

        :param api_address: Api address of nene emotion server.
        :param emotion_model_path: The path of the nene emotion mapping.
        """
        self._client = Client(api_address)
        super().__init__(api_address, emotion_mapping_path,
                         marked_emotion_mapping_path,
                         dialogue_emotion_ordering_mapping_path,
                         dialogues_emotion_mapping_path,
                         dialogues_emotion_mapping_npy_path)

    def __call__(self, text, **kwargs):
        """
        Speak the text.
        :param text: the text to speak.
        :param kwargs: the arguments for the speaker.
        :param nsfw: [required] whether the text is nsfw. Must be a boolean.
        :param emotion: [required] the emotion of the speaker. Must be a list of string.
        :return: file_path, emotion_sample
        """
        emotion = kwargs['emotion']
        result = self._client.predict(text, emotion, fn_index=2)
        message = result[0]
        if message != 'Success':
            raise SpeakerException(message)
        out_file_path = result[1]
        # copy the file to the current directory
        file_name = out_file_path.split('/')[-1]
        copy_file_path = self._out_put_path + file_name
        if not os.path.exists(copy_file_path):
            os.makedirs(os.path.dirname(copy_file_path), exist_ok=True)
        utils.copy_file(out_file_path, copy_file_path)
        return copy_file_path, emotion


class SpeakerVitsSimpleApi(SpeakerW2V2):
    def __init__(self, api_address,
                 emotion_mapping_path,
                 marked_emotion_mapping_path,
                 dialogue_emotion_ordering_mapping_path,
                 dialogues_emotion_mapping_path,
                 dialogues_emotion_mapping_npy_path):
        """
        The speaker class for vits simple api. This class is callable.
        """
        super().__init__(api_address,
                         emotion_mapping_path,
                         marked_emotion_mapping_path,
                         dialogue_emotion_ordering_mapping_path,
                         dialogues_emotion_mapping_path,
                         dialogues_emotion_mapping_npy_path)

    def __call__(self, text, id_=0, format_="wav", lang="ja", length=1, noise=0.667, noisew=0.8, max_=50, **kwargs):
        """
        Speak the text.
        :param text: the text to speak.
        :param kwargs: the arguments for the speaker.
        :param nsfw: [required] whether the text is nsfw. Must be a boolean.
        :param emotion: [required] the emotion of the speaker. Must be a list of float. The emotion is an ADV model array.
        :return: file_path, emotion_sample
        """
        if 'context' in kwargs and 'raw_text' in kwargs:
            emotion = self.get_emotion_sample_by_text(kwargs['raw_text'], kwargs['context'], text)
        elif 'nsfw' in kwargs:
            emotion = self._get_emotion_sample(kwargs['emotion'], kwargs['nsfw'])
        fields = {
            "text": text,
            "id": str(id_),
            "format": format_,
            "lang": lang,
            "length": str(length),
            "noise": str(noise),
            "noisew": str(noisew),
            "max": str(max_),
            "emotion": str(emotion)
        }
        boundary = '----VoiceConversionFormBoundary' + ''.join(random.sample(string.ascii_letters + string.digits, 16))

        m = MultipartEncoder(fields=fields, boundary=boundary)
        headers = {"Content-Type": m.content_type}
        url = f"{self._api_address}/voice/w2v2-vits"
        try:
            res = requests.post(url=url, data=m, headers=headers)
        except requests.exceptions.ConnectionError:
            time.sleep(2)
            utils.warn(f"[Vits Simple API]ConnectionError, retrying...")
            return self(text, id_, format_, lang, length, noise, noisew, max_, **kwargs)
        if res.status_code != 200:
            utils.warn(f"[Vits Simple API]Status code: {res.status_code}, please check the server.")
            return None, None
        file_name = re.findall("filename=(.+)", res.headers["Content-Disposition"])[0]
        path = f"{self._out_put_path}\\{file_name}"
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(res.content)
        return path, emotion
