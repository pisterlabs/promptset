from backend.Controller.SentenceController import EncouragedSentencesController, ProhibitedSentencesController
from backend.Model.SentenceModel import EncouragedSentenceModel, ProhibitedPhrasesModel
from backend.Model.RecordingModel import RecordingModel
from backend.Model.RequestModel import AudioGPTRequestModel
from backend.Model.RequestModel import EmbeddingRequestModel
from backend.Controller.SQLServerController import SQLSERVERDBModel, SQLServerController
from backend.Model.DB.recordingsDB import Recording, Scores
from backend.Model.jsonCreator import JsonFileCreator
from backend.Controller.PostGreSQLController import PostgreController
from backend.Controller.pathFinder import JSONFinder
from backend.Controller.PossibleWav import PossibleWav, WavModel
from backend.Controller.GPTCreator import OpenAIProxyAudio, OpenAIProxyEmbeddings
from backend.Controller.GPTCreator import OpenAIModelInterface, OpenAIEmbeddingRequest
from abc import ABC, abstractmethod
from backend.Controller.analyser import SpeechRefinement
from backend.Controller.analyser import PatternController
from backend.Controller.ScoreHandler import EncouragedPhrasesScoreHandler, ProhibitedPhrasesScoreHandler
from backend.Controller.ScoreHandler import DatabaseStoringScoreHandler
import subprocess
import json
import os


class ApplicationProcess(ABC):

    @staticmethod
    @abstractmethod
    def setup_application(y: str, m: str, d: str):
        pass

    @classmethod
    def audio_price_evaluation(cls, y: str, m: str, d: str):
        """ This method will search for all the pertinent records inside the database that were
                 recorded in a certain date the format must be y - m - d.
                 returns the price that it takes to transform all those recordings into text"""

        total_price = 0
        recordings = list[dict]()
        for line in cls.setup_application(y, m, d):
            phone_number = str(line[3])
            date = str(line[4])
            cedente = str(line[5])
            gestion_id = line[0]
            wavs = PossibleWav.get_recordings(phone_number, date, cedente)

            if wavs:
                for audio_file in wavs:
                    audio_gpt = AudioGPTRequestModel(
                        prompt="",
                        audio_path=audio_file.path,
                        name=audio_file.name,
                        size=audio_file.size
                    )

                    # We send to the database the recording data
                    audio_gpt.set_recording(gestion_id)

                    proxy_response = OpenAIProxyAudio.check_access(audio_gpt, False)
                    recordings.append(audio_file.deserialize())
                    if type(proxy_response) is float:
                        total_price += proxy_response
                JsonFileCreator.write(recordings, "../analysed_records/wav_data.json")

        return total_price


class GestionesDePago(ApplicationProcess):

    @staticmethod
    def setup_application(y: str, m: str, d: str):
        date_for_storage = {"y": y, "m": m, "d": d}
        JsonFileCreator.write(date_for_storage, "../analysed_records/date.json")
        sql_server_model = SQLSERVERDBModel()
        subprocess.call(r"C:\Users\hjimenez\Desktop\Backup\backend\openRepo.bat")
        return sql_server_model.get_all_successfull_recordings_given_date(y, m, d)

    @staticmethod
    def audio_transformation_embeddings_evaluation():
        """ This method should only be executed once the price of this transaction is calculated and the WavFiles are
         stored, it calls the OpenAI api and transforms all the records.
         It returns the price of turning these recordings into embeddings"""

        prompt = "Cliente-Alo ? Agente-Buenos Dias..."
        json_finder = JSONFinder("../analysed_records/")
        wavs_data = json_finder.find("wav_data")
        subprocess.call(r"C:\Users\hjimenez\Desktop\Backup\backend\openRepo.bat")

        total_price = 0
        for wav_json in wavs_data:
            wav_file = WavModel.serialize(wav_json)

            audio: OpenAIModelInterface = AudioGPTRequestModel(prompt, wav_file.path, wav_file.name, wav_file.size)
            audio_proxy_response = OpenAIProxyAudio.check_access(audio, True)

            embedding: OpenAIModelInterface = EmbeddingRequestModel(wav_file.name, audio_proxy_response)
            embedding_proxy_response = OpenAIProxyEmbeddings.check_access(embedding, False)

            if type(embedding_proxy_response) is float:
                total_price += embedding_proxy_response

        return total_price

    @staticmethod
    def embeddings_calculation():
        """ This method is the last method of the pattern management section.
        It should be done once all the other part are completed.
        This method takes all the recordings with their audio translated
        stored in the database of a certain date and transforms them into embeddings
        the result is stored in the database"""

        json_finder = JSONFinder("../analysed_records/")
        dates = json_finder.find('date')

        chunked_iterator_results = PostgreController.get_recordings_given_date(dates['y'], dates['m'], dates['d'])
        for row_results in chunked_iterator_results:
            recording: Recording = row_results[0]
            embedding_request = EmbeddingRequestModel(recording.name, recording.audio_text)
            proxy_response = OpenAIProxyEmbeddings.check_access(embedding_request, True)
            print("Proxy Response : ", proxy_response)

        return True


class QualityAssurance(ApplicationProcess):

    @staticmethod
    def setup_application(y: str, m: str, d: str):
        date_for_storage = {"y": y, "m": m, "d": d}
        JsonFileCreator.write(date_for_storage, "../analysed_records/date.json")
        sql_server_model = SQLSERVERDBModel()
        subprocess.call(r"C:\Users\hjimenez\Desktop\Backup\backend\openRepo.bat")
        return sql_server_model.get_all_recordings_given_date(y, m, d)

    @staticmethod
    def audio_transformation_score_calculation():
        """ This method  transforms the stored recording data and transforms it into text,
        this method should always be executed after audio_price_evaluation,
        once the audio is turned into text it is evaluated with certain criteria,
         specified throw the SQLServer database """

        prompt = ""
        json_finder = JSONFinder("../analysed_records")
        wavs_data = json_finder.find("wav_data")

        for wav_json in wavs_data:
            wav_model = WavModel.serialize(wav_json)

            # Audio transformation into text with the help of a Proxy
            audio_request = AudioGPTRequestModel(prompt, wav_model.path, wav_model.name, wav_model.size)
            proxy_response = OpenAIProxyAudio.check_access(audio_request, True)

            # Score Calculation with chain of responsibility handlers
            encouraged = EncouragedPhrasesScoreHandler()
            prohibited = ProhibitedPhrasesScoreHandler()
            database_storing = DatabaseStoringScoreHandler()
            encouraged.set_next(prohibited).set_next(database_storing)

            recording = PostgreController.get_recording(wav_model.name)
            data = {"r_id": recording[0].id}
            encouraged.handle(EncouragedSentenceModel(proxy_response, wav_model.cedente), data)

        return True
    @staticmethod
    def await_test(request_values):
        date = dict()
        for date_string in request_values.form:
            date = json.loads(date_string)

        print(date)
        scores_given_date = PostgreController.get_scores_given_date(date["year"], date["month"], date["day"])

        processed_view = dict()
        for score_row in scores_given_date:
            score: Scores = score_row[0]
            processed_view[score.s_id] = score.score
        return processed_view

