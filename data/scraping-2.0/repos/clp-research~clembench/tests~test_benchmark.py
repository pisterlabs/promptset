import unittest

from clemgame import benchmark, string_utils
from backends import openai_api, alephalpha_api


class BenchmarkTestCase(unittest.TestCase):
    def test_list_games(self):
        benchmark.list_games()

    def test_run_taboo(self):
        benchmark.run(game_name="taboo", temperature=0.0,
                      experiment_name="low_en",
                      dialog_pair=string_utils.to_pair_descriptor([openai_api.MODEL_GPT_35, openai_api.MODEL_GPT_35]))

    def test_run_taboo_human(self):
        benchmark.run(game_name="taboo", temperature=0.0,
                      experiment_name="low_en",
                      dialog_pair=string_utils.to_pair_descriptor(["human", "human"]))

    def test_transcribe_privateshared(self):
        benchmark.transcripts(game_name="privateshared")

    def test_score_taboo(self):
        benchmark.score(game_name="taboo")

    def test_hello_game(self):
        # Only run specific experiment
        benchmark.run(game_name="hellogame", temperature=1.0,
                      experiment_name="greet_en", dialog_pair=alephalpha_api.LUMINOUS_SUPREME_CONTROL)

    def test_run_privateshared(self):
        benchmark.run(game_name="privateshared", temperature=0.0, dialog_pair="default")

    def test_run_imagegame(self):
        benchmark.run(game_name="imagegame", temperature=0.0,
                      experiment_name="compact_grids",
                      dialog_pair=string_utils.to_pair_descriptor([openai_api.MODEL_GPT_3, openai_api.MODEL_GPT_3]))

    def test_run_referencegame(self):
        benchmark.run(game_name="referencegame", temperature=0.0,
                      experiment_name="hard_grids_edit_distance_2",
                      dialog_pair=string_utils.to_pair_descriptor([openai_api.MODEL_GPT_3, openai_api.MODEL_GPT_3]))

    def test_run_wordle(self):
        benchmark.run(game_name="wordle", temperature=0.0,
                      experiment_name="low_frequency_words_no_clue_no_critic",
                      dialog_pair=string_utils.to_pair_descriptor([openai_api.MODEL_GPT_35, openai_api.MODEL_GPT_35]))

    def test_run_all(self):
        benchmark.run(game_name="all", dialog_pair="dry_run")


if __name__ == '__main__':
    unittest.main()
