from backend.Model.DB.base import engine, Session
from backend.Model.DB.recordingsDB import Base, Recording, Embedding, Scores
from backend.Model.RecordingModel import RecordingModel
from backend.Model.DB.SQLServerModel import SQLSERVERDBModel
from backend.Controller.SentenceController import EncouragedSentencesController
from backend.Controller.SentenceController import ProhibitedSentencesController
from backend.Model.SentenceModel import EncouragedSentenceModel, ProhibitedPhrasesModel
from backend.Controller.analyser import SpeechRefinement
from backend.Controller.pathFinder import JSONFinder
from backend.Controller.PossibleWav import PossibleWav
from backend.Model.RequestModel import OpenAIModelInterface, ChatGPTRequestModel, AudioGPTRequestModel
from backend.Controller.GPTCreator import OpenAIProxy, OpenAIProxyAudio, OpenAIProxyEmbeddings
from backend.Controller.ScoreHandler import EncouragedPhrasesScoreHandler, ProhibitedPhrasesScoreHandler
from backend.Controller.ScoreHandler import DatabaseStoringScoreHandler
from sqlalchemy import text, select
import subprocess
import unittest
import shutil
import os


class FlaskTesting:
    @staticmethod
    def make_post_simple():
        url = "http://127.0.0.1:5000/records"

        audio_text = {"content": "Alo ?"}
        score = {"QA": "Dos tablas "}


class ProxyPatternTests(unittest.TestCase):
    def test_proxy_pattern_already_existing(self):
        controller = SQLSERVERDBModel()
        prompt = "Cliente-Alo ? Agente-Buenos Dias"
        subprocess.call(r"C:\Users\hjimenez\Desktop\Backup\backend\openRepo.bat")
        for line in controller.test():
            final_wavs = PossibleWav.get_recordings(str(line[3]), str(line[4]), str(line[5]))
            if final_wavs is not None:
                print(line)
                for final_wav in final_wavs:

                    audio = AudioGPTRequestModel(prompt, final_wav.path, final_wav.name, final_wav.size)
                    diarized_test = OpenAIProxyAudio.operation(audio, False)

                    print(diarized_test)

                    self.assertTrue(type(diarized_test) is str)

                    self.assertTrue(len(diarized_test) > 50)

    def test_proxy_pattern_unexisting(self):
        path = '../analysed_records/audio_text'
        if os.path.exists(path):
            self._delete_folder_with_content(path)
        self.test_proxy_pattern_already_existing()

    @staticmethod
    def _delete_folder_with_content(path_to_folder):
        folder = path_to_folder
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        os.rmdir(path_to_folder)


class ChainOfResponsibilityTest(unittest.TestCase):

    def test_chain_follows_the_order_specified(self):
        encouraged = EncouragedPhrasesScoreHandler()
        prohibited = ProhibitedPhrasesScoreHandler()
        database_storage = DatabaseStoringScoreHandler()

        encouraged.set_next(prohibited).set_next(database_storage)

        encouraged_model = EncouragedSentenceModel("Buenos Dias", "Unexistent")

        recording = RecordingModel("chainOfResponsibilityTest")
        recording.set_recording("12356")
        recording_db = recording.get_recording_row()

        result: Scores = encouraged.handle(encouraged_model, {"r_id": recording_db[0].id})

        self.assertEqual(type(result), Scores)

        print(result.s_id, result.score)

    def test_chain_gives_an_error_if_wrong_order(self):
        encouraged = EncouragedPhrasesScoreHandler()
        prohibited = ProhibitedPhrasesScoreHandler()
        database_storage = DatabaseStoringScoreHandler()

        encouraged_model = EncouragedSentenceModel("Buenos Dias", "Unexistent")

        recording = RecordingModel("chainOfResponsibilityTest")
        recording.set_recording("12356")
        recording_db: Recording = recording.get_recording_row()[0]

        encouraged.set_next(database_storage).set_next(prohibited)

        with self.assertRaises(AttributeError):
            encouraged.handle(encouraged_model, {"r_id": recording_db.id})

        prohibited.set_next(encouraged).set_next(database_storage).set_next(None)

        prohibited_model = ProhibitedPhrasesModel("Buenos Dias")
        with self.assertRaises(AttributeError):

            prohibited.handle(
                prohibited_model, {"ticket_positive": {"CEDENTE": 9}, "r_id": recording_db.id, "positive": 9}
            )

