import sys
sys.path.append('/Users/tschip/workspace/baa/baa-ruefer/')
#sys.path.append('/home/jovyan/baa-ruefer/')
import numpy as np
from nltk import ngrams
from difflib import SequenceMatcher
import openai
import json
import time
import re
import tqdm
from rouge import Rouge

from data_loaders.JSONDataLoader import JSONDataLoader


class LlamaGenerate():
    def __init__(self, team_ids: list, requests_api=None):
        dataloader = JSONDataLoader(team_ids)
        self.team_ids = team_ids
        self.player_names = dataloader.get_player_names()
        self.player_information = dataloader.get_player_information()
        self.player_statistics = dataloader.get_player_statistics()
        self.player_injuries = dataloader.get_player_injuries()
        self.team_information = dataloader.get_team_information()
        self.venue_information = dataloader.get_venue_information()
        self.team_statistics = dataloader.get_team_statistics()
        self.player_transfers = dataloader.get_player_transfers()
        self.player_news = dataloader.get_player_news()
        self.news_details = dataloader.get_news_details()
        self.team_news = dataloader.get_team_news()
        self.head_to_head = dataloader.get_head_to_head()
        self.home_team = dataloader.get_home_team_id()
        self.players_ids = dataloader.get_players_ids()
        self.team_fixtures = dataloader.get_team_fixtures()
        self.todays_fixture_id = dataloader.get_todays_fixture_id()
        self.fixture_lineup = dataloader.get_fixture_lineup()
        self.fixture_statistics = dataloader.get_fixture_stats()
        self.coaches = dataloader.get_coaches()

        self.team_information_parentT = []
        self.team_statistics_parentT = []
        self.team_injuries_parentT = []
        self.venue_information_parentT = []
        self.player_information_parentT = []
        self.player_statistics_parentT = []
        self.player_transfers_parentT = []
        self.fixture_information_parentT = []
        self.fixture_statistics_parentT = []
        self.coach_parentT = []
        self.rouge_scores_1 = []
        self.rouge_scores_2 = []
        self.rouge_scores_l = []
    
    def get_players_ids(self):
        return self.players_ids
    
    def get_team_ids(self):
        return self.team_ids
    
    def get_player_names(self):
        return self.player_names
    
    def get_team_name(self, team_id):
        for fixture_key in dict(sorted(self.head_to_head.items())):
            if self.head_to_head[fixture_key]['home_id'] == team_id:
                return self.head_to_head[fixture_key]['home_name']
            else:
                return self.head_to_head[fixture_key]['away_name']
            
    def get_team_statistics(self, team_id):
        return self.team_statistics[team_id]
    
    def get_team_information_parentT(self):
        return self.team_information_parentT[-1]
    
    def get_player_information_parentT(self):
        return self.player_information_parentT[-1]
    
    def get_venue_information_parentT(self):
        return self.venue_information_parentT[-1]
    
    def get_fixture_information_parentT(self):
        return self.fixture_information_parentT[-1]
    
    def get_overall_team_information_parentT(self):
        return sum(self.team_information_parentT) / len(self.team_information_parentT) if len(self.team_information_parentT) > 0 else None

    def get_overall_team_statistics_parentT(self):
        return sum(self.team_statistics_parentT) / len(self.team_statistics_parentT) if len(self.team_statistics_parentT) > 0 else None
    
    def get_overall_team_injuries_parentT(self):
        return sum(self.team_injuries_parentT) / len(self.team_injuries_parentT) if len(self.team_injuries_parentT) > 0 else None

    def get_overall_player_information_parentT(self):
        return sum(self.player_information_parentT) / len(self.player_information_parentT) if len(self.player_information_parentT) > 0 else None
    
    def get_overall_player_statistics_parentT(self):
        return sum(self.player_statistics_parentT) / len(self.player_statistics_parentT) if len(self.player_statistics_parentT) > 0 else None
    
    def get_overall_player_transfers_parentT(self):
        return sum(self.player_transfers_parentT) / len(self.player_transfers_parentT) if len(self.player_transfers_parentT) > 0 else None
    
    def get_overall_venue_information_parentT(self):
        return sum(self.venue_information_parentT) / len(self.venue_information_parentT) if len(self.venue_information_parentT) > 0 else None
    
    def get_overall_fixture_information_parentT(self):
        return sum(self.fixture_information_parentT) / len(self.fixture_information_parentT) if len(self.fixture_information_parentT) > 0 else None
    
    def get_overall_fixture_statistics_parentT(self):
        return sum(self.fixture_statistics_parentT) / len(self.fixture_statistics_parentT) if len(self.fixture_statistics_parentT) > 0 else None
    
    def get_overall_coach_parentT(self):
        return sum(self.coach_parentT) / len(self.coach_parentT) if len(self.coach_parentT) > 0 else None
    
    def get_overall_rouge_scores(self):
        return (sum(self.rouge_scores_1) / len(self.rouge_scores_1)) if len(self.rouge_scores_1) > 0 else None, (sum(self.rouge_scores_2) / len(self.rouge_scores_2)) if len(self.rouge_scores_2) > 0 else None, (sum(self.rouge_scores_l) / len(self.rouge_scores_l)) if len(self.rouge_scores_l) > 0 else None
    
    def generate_team_information(self, team_id):
        messages = [
            {   
                "role": "system", 
                "content": "You are a sports journalist for football (soccer) respond in sentences that can be used in live commentary. Just return the sentence. Without introduction what you did."
            },
            {
                "role": "user",
                "content": "Generate a sentence about the team information. The informations are provided in the JSON. {'name': 'Lazio', 'code': 'LAZ', 'country': 'Italy', 'founded': 1900}"
            },
            {
                "role": "assistant",
                "content": "Lazio, proudly bearing the code LAZ, hails from the heart of Italy. With roots tracing back to 1900, they have a rich history in the panorama of world football, steadily leaving their mark through every passing decade."
            },
            {
                "role": "user",
                "content": f"Generate a sentence about the team information. The informations are provided in the JSON. {self.get_team_information()[team_id]}"
            },
        ]
        model_outputs = self._generate_llama_output(messages)

        li = []
        for item in self.get_team_information()[team_id].items():
            for i in str(item[1]).split(' '):
                li.append(i)

        self.team_information_parentT.append(self.parent_t_score(model_outputs, li))

        return model_outputs
    
    def generate_team_statistics(self, team_id):
        messages = [
            {   
                "role": "system", 
                "content": "You are a sports journalist for football (soccer) respond in sentences that can be used in live commentary. Just return the sentences about statistical facts in JSON format. Without introduction what you did."
            },
            {
                "role": "user",
                "content": """Give me multiple different sentences about the team statistics. The informations are provided in the JSON. Return the sentences as a python dictionary. For each fact create a key with the topic and the sentecnes as value. Generate text without \ and '\n'. {'team_name': 'Lazio', 'fixtures_played_home': 19, 'fixtures_played_away': 19, 'fixtures_wins_home': 10, 'fixtures_wins_away': 8, 'fixtures_draws_home': 6, 'fixtures_draws_away': 4, 'fixtures_loses_home': 3, 'fixtures_loses_away': 7, 'fixtures_loses_total': 10, 'goals_for_total_home': 42, 'goals_for_total_away': 35, 'goals_for_average_home': '2.2', 'goals_for_average_away': '1.8', 'goals_for_average_total': '2.0', 'goals_for_minute_0-15_total': 12, 'goals_for_minute_0-15_percentage': '16.00%', 'goals_for_minute_16-30_total': 7, 'goals_for_minute_16-30_percentage': '9.33%', 'goals_for_minute_31-45_total': 14, 'goals_for_minute_31-45_percentage': '18.67%', 'goals_for_minute_46-60_total': 13, 'goals_for_minute_46-60_percentage': '17.33%', 'goals_for_minute_61-75_total': 15, 'goals_for_minute_61-75_percentage': '20.00%', 'goals_for_minute_76-90_total': 8, 'goals_for_minute_76-90_percentage': '10.67%', 'goals_for_minute_91-105_total': 6,
                 'goals_for_minute_91-105_percentage': '8.00%', 'goals_for_minute_106-120_total': None, 'goals_for_minute_106-120_percentage': None, 'goals_against_total_home': 25, 'goals_against_total_away': 33, 'goals_against_total_total': 58, 'goals_against_average_home': '1.3', 'goals_against_average_away': '1.7', 'goals_against_average_total': '1.5', 'goals_against_minute_0-15_total': 14, 'goals_against_minute_0-15_percentage': '23.33%', 'goals_against_minute_16-30_total': 8, 'goals_against_minute_16-30_percentage': '13.33%', 'goals_against_minute_31-45_total': 8,
                 'goals_against_minute_31-45_percentage': '13.33%', 'goals_against_minute_46-60_total': 7, 'goals_against_minute_46-60_percentage': '11.67%', 'goals_against_minute_61-75_total': 10, 'goals_against_minute_61-75_percentage': '16.67%', 'goals_against_minute_76-90_total': 7, 'goals_against_minute_76-90_percentage': '11.67%', 'goals_against_minute_91-105_total': 6, 'goals_against_minute_91-105_percentage': '10.00%'}
                        """
            },
            {
                "role": "assistant",
                "content": """{
                \"fixtures_played\": \"Lazio has played an equal amount of games at home and away this season, with a total of 19 each, summing up to a total of 38 fixtures overall.\",
                \"fixtures_wins\": \"In terms of victories, Lazio has a slight home advantage, with 10 wins at home and 8 on the road, bringing their total fixtures wins to 18.\",
                \"fixtures_draws_loses\": \"Lazio has had 10 draws and 10 losses in total this season, with a marginally higher draws and losses count at home.\",
                \"goal_statistics\": \"The team scored a total of 77 goals, with 42 at home and 35 away, which gives them an average of 2.0 goals per game. Regarding conceded goals, they let in 58 goals in total, slightly more away (33) than at home (25).\",
                \"goal_timing\": \"Lazio tends to score most of their goals between the 61st and 75th minute, while they are most vulnerable to conceding in the opening 15 minutes of the game.\",
                
                }"""
            },
            {
                "role": "user",
                "content": f"Give me multiple different sentences about the team statistics. The informations are provided in the JSON. Return the sentences as a python dictionary. For each fact create a key with the topic and the sentecnes as value. Generate text without \ and '\n'. {self.team_statistics[team_id]}"
            },
        ]
        team_statistics_output = self._generate_llama_output(messages)
        try:
            model_outputs = json.loads(self.correct_json(self.extract_first_match(team_statistics_output)))
            output_strings = []
            for statistic in model_outputs:
                output_strings.append(model_outputs[statistic])

            li = []
            for item in self.get_team_statistics(team_id).items():
                for i in str(item[1]).split(' '):
                    li.append(i)

            self.team_statistics_parentT.append(self.system_level_parent_t_score(output_strings, li))
            return model_outputs
        
        except:
            self.team_statistics_parentT.append(0)
            return None
            
    def generate_team_news(self, team_id):
        teams = {}
        news = self.get_team_news()
        teams[team_id] = {}

        if str(team_id) in news:        
            teams[team_id]['news'] = {}
            for n in news[str(team_id)].items():
                pattern = re.compile('<.*?>')
                result = re.sub(pattern, '', n[1])
                teams[team_id]['news'][n[0]] = self._generate_llama_output(f"Summarize the news article with focus on the team {self.get_team_name(team_id)}. The article is provided in the JSON. Format it as a normal text. summarize it with at most 3 sentences. {result}")
                self.calculate_rouge_scores(teams[team_id]['news'][n[0]], result)
                break
        else:
            teams[team_id]['news'] = None

        return teams

    def generate_team_injuries(self, team_id):
        fixture_id = self.todays_fixture_id
        injuries = self.player_injuries[str(team_id)]
        injuries_dict = {}
        outputs = {}
        print(f"Generate players injuries for team {team_id}")
        for player_id in tqdm.tqdm(injuries):
            if injuries[player_id]['fixture_id'] == fixture_id:
                injuries_dict[player_id] = injuries[player_id]
                injuries_dict[player_id]['player_name'] = self.get_player_names()[team_id][player_id]
                del injuries_dict[player_id]['fixture_id']
                messages = [
                    {
                        "role": "system",
                        "content": "You are a sports journalist for football (soccer) respond in sentences that can be used in live commentary. Just return the sentence. Without introduction what you did."
                    },
                    {
                        "role": "user",
                        "content": f"Generate a sentence about the player and the injury for the fixture. The information are provided in the JSON. {injuries_dict[player_id]}"
                    },
                ]
                outputs[player_id] = self._generate_llama_output(messages)
                li = []
                for item in injuries_dict[player_id]:
                    for i in str(item[1]).split(' '):
                        li.append(i)

                self.team_injuries_parentT.append(self.parent_t_score(outputs[player_id], li))

        return outputs

    def generate_team_players(self, team_id):
        players = {}
        information = self.get_player_information()[str(team_id)]
        statistics = self.get_player_statistics()[str(team_id)]
        transfers = self.get_player_transfers()
        transfer_dict = {}
        news = self.get_player_news()
        counter = 1
        print(f"Generating player information and statistics for team team_id")
        for player_id in tqdm.tqdm(self.get_player_ids_from_fixture(team_id)):
            players[player_id] = {}
            try:
                del information[player_id]['injured']
                del information[player_id]['name']
            except:
                pass
            player_information_message = [
                {   
                    "role": "system", 
                    "content": "You are a sports journalist for football (soccer) respond in sentences that can be used in live commentary. Just return the sentence. Without introduction what you did."
                },
                {
                    "role": "user",
                    "content": "Generate a sentence about the team information. The informations are provided in the JSON. {'name': 'Lazio', 'code': 'LAZ', 'country': 'Italy', 'founded': 1900}"
                },
                {
                    "role": "assistant",
                    "content": "Lazio, proudly bearing the code LAZ, hails from the heart of Italy. With roots tracing back to 1900, they have a rich history in the panorama of world football, steadily leaving their mark through every passing decade."
                },
                {
                    "role": "user",
                    "content": f"Generate a sentence about the player information. The informations are provided in the JSON. {information[player_id]}"
                },
            ]
            player_information_output = self._generate_llama_output(player_information_message)
            try:
                model_outputs = json.loads(self.correct_json(self.extract_first_match(player_information_output)))
                output_string = ''
                for item in model_outputs:
                    output_string += model_outputs[item]

                li = []
                for item in information[player_id]:
                    for i in str(item[1]).split(' '):
                        li.append(i)

                self.player_information_parentT.append(self.parent_t_score(output_string, li))

                players[player_id]['information'] = model_outputs
            except:
                self.player_information_parentT.append(0)
                players[player_id]['information'] = None

            statistics[player_id]['player_name'] = self.get_player_names()[team_id][player_id]
            player_statistics_messages = [
                {   
                    "role": "system", 
                    "content": "You are a sports journalist for football (soccer) respond in sentences that can be used in live commentary. Just return the sentences about statistical facts in JSON format. Without introduction what you did."
                },
                {
                    "role": "user",
                    "content": """Give me multiple different sentences about the team statistics. The informations are provided in the JSON. Return the sentences as a python dictionary. For each fact create a key with the topic and the sentecnes as value. Generate text without \ and '\n'. {'team_name': 'Lazio', 'fixtures_played_home': 19, 'fixtures_played_away': 19, 'fixtures_wins_home': 10, 'fixtures_wins_away': 8, 'fixtures_draws_home': 6, 'fixtures_draws_away': 4, 'fixtures_loses_home': 3, 'fixtures_loses_away': 7, 'fixtures_loses_total': 10, 'goals_for_total_home': 42, 'goals_for_total_away': 35, 'goals_for_average_home': '2.2', 'goals_for_average_away': '1.8', 'goals_for_average_total': '2.0', 'goals_for_minute_0-15_total': 12, 'goals_for_minute_0-15_percentage': '16.00%', 'goals_for_minute_16-30_total': 7, 'goals_for_minute_16-30_percentage': '9.33%', 'goals_for_minute_31-45_total': 14, 'goals_for_minute_31-45_percentage': '18.67%', 'goals_for_minute_46-60_total': 13, 'goals_for_minute_46-60_percentage': '17.33%', 'goals_for_minute_61-75_total': 15, 'goals_for_minute_61-75_percentage': '20.00%', 'goals_for_minute_76-90_total': 8, 'goals_for_minute_76-90_percentage': '10.67%', 'goals_for_minute_91-105_total': 6,
                    'goals_for_minute_91-105_percentage': '8.00%', 'goals_for_minute_106-120_total': None, 'goals_for_minute_106-120_percentage': None, 'goals_against_total_home': 25, 'goals_against_total_away': 33, 'goals_against_total_total': 58, 'goals_against_average_home': '1.3', 'goals_against_average_away': '1.7', 'goals_against_average_total': '1.5', 'goals_against_minute_0-15_total': 14, 'goals_against_minute_0-15_percentage': '23.33%', 'goals_against_minute_16-30_total': 8, 'goals_against_minute_16-30_percentage': '13.33%', 'goals_against_minute_31-45_total': 8,
                    'goals_against_minute_31-45_percentage': '13.33%', 'goals_against_minute_46-60_total': 7, 'goals_against_minute_46-60_percentage': '11.67%', 'goals_against_minute_61-75_total': 10, 'goals_against_minute_61-75_percentage': '16.67%', 'goals_against_minute_76-90_total': 7, 'goals_against_minute_76-90_percentage': '11.67%', 'goals_against_minute_91-105_total': 6, 'goals_against_minute_91-105_percentage': '10.00%'}
                            """
                },
                {
                    "role": "assistant",
                    "content": """{
                    \"fixtures_played\": \"Lazio has played an equal amount of games at home and away this season, with a total of 19 each, summing up to a total of 38 fixtures overall.\",
                    \"fixtures_wins\": \"In terms of victories, Lazio has a slight home advantage, with 10 wins at home and 8 on the road, bringing their total fixtures wins to 18.\",
                    \"fixtures_draws_loses\": \"Lazio has had 10 draws and 10 losses in total this season, with a marginally higher draws and losses count at home.\",
                    \"goal_statistics\": \"The team scored a total of 77 goals, with 42 at home and 35 away, which gives them an average of 2.0 goals per game. Regarding conceded goals, they let in 58 goals in total, slightly more away (33) than at home (25).\",
                    \"goal_timing\": \"Lazio tends to score most of their goals between the 61st and 75th minute, while they are most vulnerable to conceding in the opening 15 minutes of the game.\",
                    
                    }"""
                },
                {
                    "role": "user",
                    "content": f"Give me multiple different sentences about the player statistics. The informations are provided in the JSON. Return the sentences as a python dictionary. For each fact create a key with the topic and the sentecnes as value. Generate text without \ and '\n'. {statistics[player_id]}"
                },
            ]

            player_statistics_output = self._generate_llama_output(player_statistics_messages)
            try:
                model_outputs = json.loads(self.correct_json(self.extract_first_match(player_statistics_output)))
                output_strings = []
                for statistic in model_outputs:
                    output_strings.append(model_outputs[statistic])

                li = []
                for item in statistics[player_id].items():
                    for i in str(item[1]).split(' '):
                        li.append(i)

                self.player_statistics_parentT.append(self.system_level_parent_t_score(output_strings, li))
                players[player_id]['statistics'] = model_outputs

            except:
                self.player_statistics_parentT.append(0)
                players[player_id]['statistics'] =  None

            try:
                transfer_dict[player_id] = {}
                transfer_dict[player_id]['player_name'] = self.get_player_names()[team_id][player_id]
                transfer_dict[player_id]['transfers'] = transfers[str(player_id)]
                player_transfers_message = [
                    {   
                        "role": "system", 
                        "content": "You are a sports journalist for football (soccer) respond in sentences that can be used in live commentary. Just return the sentences about statistical facts in JSON format. Without introduction what you did."
                    },
                    {
                        "role": "user",
                        "content": """Give me multiple different sentences about the team statistics. The informations are provided in the JSON. Return the sentences as a python dictionary. For each fact create a key with the topic and the sentecnes as value. Generate text without \ and '\n'. {'team_name': 'Lazio', 'fixtures_played_home': 19, 'fixtures_played_away': 19, 'fixtures_wins_home': 10, 'fixtures_wins_away': 8, 'fixtures_draws_home': 6, 'fixtures_draws_away': 4, 'fixtures_loses_home': 3, 'fixtures_loses_away': 7, 'fixtures_loses_total': 10, 'goals_for_total_home': 42, 'goals_for_total_away': 35, 'goals_for_average_home': '2.2', 'goals_for_average_away': '1.8', 'goals_for_average_total': '2.0', 'goals_for_minute_0-15_total': 12, 'goals_for_minute_0-15_percentage': '16.00%', 'goals_for_minute_16-30_total': 7, 'goals_for_minute_16-30_percentage': '9.33%', 'goals_for_minute_31-45_total': 14, 'goals_for_minute_31-45_percentage': '18.67%', 'goals_for_minute_46-60_total': 13, 'goals_for_minute_46-60_percentage': '17.33%', 'goals_for_minute_61-75_total': 15, 'goals_for_minute_61-75_percentage': '20.00%', 'goals_for_minute_76-90_total': 8, 'goals_for_minute_76-90_percentage': '10.67%', 'goals_for_minute_91-105_total': 6,
                        'goals_for_minute_91-105_percentage': '8.00%', 'goals_for_minute_106-120_total': None, 'goals_for_minute_106-120_percentage': None, 'goals_against_total_home': 25, 'goals_against_total_away': 33, 'goals_against_total_total': 58, 'goals_against_average_home': '1.3', 'goals_against_average_away': '1.7', 'goals_against_average_total': '1.5', 'goals_against_minute_0-15_total': 14, 'goals_against_minute_0-15_percentage': '23.33%', 'goals_against_minute_16-30_total': 8, 'goals_against_minute_16-30_percentage': '13.33%', 'goals_against_minute_31-45_total': 8,
                        'goals_against_minute_31-45_percentage': '13.33%', 'goals_against_minute_46-60_total': 7, 'goals_against_minute_46-60_percentage': '11.67%', 'goals_against_minute_61-75_total': 10, 'goals_against_minute_61-75_percentage': '16.67%', 'goals_against_minute_76-90_total': 7, 'goals_against_minute_76-90_percentage': '11.67%', 'goals_against_minute_91-105_total': 6, 'goals_against_minute_91-105_percentage': '10.00%'}
                                """
                    },
                    {
                        "role": "assistant",
                        "content": """{
                        \"fixtures_played\": \"Lazio has played an equal amount of games at home and away this season, with a total of 19 each, summing up to a total of 38 fixtures overall.\",
                        \"fixtures_wins\": \"In terms of victories, Lazio has a slight home advantage, with 10 wins at home and 8 on the road, bringing their total fixtures wins to 18.\",
                        \"fixtures_draws_loses\": \"Lazio has had 10 draws and 10 losses in total this season, with a marginally higher draws and losses count at home.\",
                        \"goal_statistics\": \"The team scored a total of 77 goals, with 42 at home and 35 away, which gives them an average of 2.0 goals per game. Regarding conceded goals, they let in 58 goals in total, slightly more away (33) than at home (25).\",
                        \"goal_timing\": \"Lazio tends to score most of their goals between the 61st and 75th minute, while they are most vulnerable to conceding in the opening 15 minutes of the game.\",
                        
                        }"""
                    },
                    {
                        "role": "user",
                        "content": f"Give me multiple different sentences about the player transfers. The informations are provided in the JSON. Return the sentences as a python dictionary. For each fact create a key with the topic and the sentecnes as value. Generate text without \ and '\n'. {transfer_dict[player_id]}"
                    },
                ]

                player_transfers = self._generate_llama_output(player_transfers_message)
                try:
                    model_outputs = json.loads(self.correct_json(self.extract_first_match(player_transfers)))
                    output_strings = []
                    for transfer in model_outputs.items():
                        output_strings.append(transfer[1])

                    li = []
                    for item in transfer_dict[player_id].items():
                        for i in str(item[1]).split(' '):
                            li.append(i)

                    self.player_transfers_parentT.append(self.system_level_parent_t_score(output_strings, li))

                    players[player_id]['transfers'] = json.loads(self.correct_json(self.extract_first_match(player_transfers)))
                except:
                    self.player_transfers_parentT.append(0)
                    players[player_id]['transfers'] = None
            except:
                players[player_id]['transfers'] = 'No transfers for this player'

            if str(player_id) in news:
                players[player_id]['news'] = {}
                for n in news[str(player_id)].items():
                    pattern = re.compile('<.*?>')
                    result = re.sub(pattern, '', n[1])
                    players[player_id]['news'][n[0]] = self._generate_llama_output(f"""Summarize the news article with focus on the player {self.get_player_names()[team_id][player_id]}. 
                                                                                   Summarize it with at most 3 sentences. {result}""")
                    self.calculate_rouge_scores(players[player_id]['news'][n[0]], result)
                    break
            else:
                players[player_id]['news'] = None
            counter += 1
            if counter % 10 == 0:
                time.sleep(3)
        return players

    def generate_team_coach(self, team_id):
        team_coach_messages = [
                {   
                    "role": "system", 
                    "content": "You are a sports journalist for football (soccer) respond in sentences that can be used in live commentary. Just return the sentences about statistical facts in JSON format. Without introduction what you did."
                },
                {
                    "role": "user",
                    "content": """Give me multiple different sentences about the team statistics. The informations are provided in the JSON. Return the sentences as a python dictionary. For each fact create a key with the topic and the sentecnes as value. Generate text without \ and '\n'. {'team_name': 'Lazio', 'fixtures_played_home': 19, 'fixtures_played_away': 19, 'fixtures_wins_home': 10, 'fixtures_wins_away': 8, 'fixtures_draws_home': 6, 'fixtures_draws_away': 4, 'fixtures_loses_home': 3, 'fixtures_loses_away': 7, 'fixtures_loses_total': 10, 'goals_for_total_home': 42, 'goals_for_total_away': 35, 'goals_for_average_home': '2.2', 'goals_for_average_away': '1.8', 'goals_for_average_total': '2.0', 'goals_for_minute_0-15_total': 12, 'goals_for_minute_0-15_percentage': '16.00%', 'goals_for_minute_16-30_total': 7, 'goals_for_minute_16-30_percentage': '9.33%', 'goals_for_minute_31-45_total': 14, 'goals_for_minute_31-45_percentage': '18.67%', 'goals_for_minute_46-60_total': 13, 'goals_for_minute_46-60_percentage': '17.33%', 'goals_for_minute_61-75_total': 15, 'goals_for_minute_61-75_percentage': '20.00%', 'goals_for_minute_76-90_total': 8, 'goals_for_minute_76-90_percentage': '10.67%', 'goals_for_minute_91-105_total': 6,
                    'goals_for_minute_91-105_percentage': '8.00%', 'goals_for_minute_106-120_total': None, 'goals_for_minute_106-120_percentage': None, 'goals_against_total_home': 25, 'goals_against_total_away': 33, 'goals_against_total_total': 58, 'goals_against_average_home': '1.3', 'goals_against_average_away': '1.7', 'goals_against_average_total': '1.5', 'goals_against_minute_0-15_total': 14, 'goals_against_minute_0-15_percentage': '23.33%', 'goals_against_minute_16-30_total': 8, 'goals_against_minute_16-30_percentage': '13.33%', 'goals_against_minute_31-45_total': 8,
                    'goals_against_minute_31-45_percentage': '13.33%', 'goals_against_minute_46-60_total': 7, 'goals_against_minute_46-60_percentage': '11.67%', 'goals_against_minute_61-75_total': 10, 'goals_against_minute_61-75_percentage': '16.67%', 'goals_against_minute_76-90_total': 7, 'goals_against_minute_76-90_percentage': '11.67%', 'goals_against_minute_91-105_total': 6, 'goals_against_minute_91-105_percentage': '10.00%'}
                            """
                },
                {
                    "role": "assistant",
                    "content": """{
                    \"fixtures_played\": \"Lazio has played an equal amount of games at home and away this season, with a total of 19 each, summing up to a total of 38 fixtures overall.\",
                    \"fixtures_wins\": \"In terms of victories, Lazio has a slight home advantage, with 10 wins at home and 8 on the road, bringing their total fixtures wins to 18.\",
                    \"fixtures_draws_loses\": \"Lazio has had 10 draws and 10 losses in total this season, with a marginally higher draws and losses count at home.\",
                    \"goal_statistics\": \"The team scored a total of 77 goals, with 42 at home and 35 away, which gives them an average of 2.0 goals per game. Regarding conceded goals, they let in 58 goals in total, slightly more away (33) than at home (25).\",
                    \"goal_timing\": \"Lazio tends to score most of their goals between the 61st and 75th minute, while they are most vulnerable to conceding in the opening 15 minutes of the game.\",
                    
                    }"""
                },
                {
                    "role": "user",
                    "content": f"Give me multiple different sentences about the coach. The informations are provided in the JSON. Return the sentences as a python dictionary. For each fact create a key with the topic and the sentecnes as value. Generate text without \ and '\n'. {self.coaches[str(team_id)]}"
                },
            ]
        team_coach_output = self._generate_llama_output(team_coach_messages)
        try:
            model_outputs = json.loads(self.correct_json(self.extract_first_match(team_coach_output)))
            output_strings = []
            for statistic in model_outputs:
                output_strings.append(model_outputs[statistic])

            li = []
            for item in self.coaches[str(team_id)].items():
                for i in str(item[1]).split(' '):
                    li.append(i)

            self.coach_parentT.append(self.system_level_parent_t_score(output_strings, li))

            return model_outputs
        except:
            self.coach_parentT.append(0)
            return None

    def generate_fixture(self):
        fixture = {}
        information = self.get_fixture_information()
        statistics = self.get_fixture_statistics()

        fixture_information_message = [
                {   
                    "role": "system", 
                    "content": "You are a sports journalist for football (soccer) respond in sentences that can be used in live commentary. Just return the sentence. Without introduction what you did."
                },
                {
                    "role": "user",
                    "content": "Generate a sentence about the team information. The informations are provided in the JSON. {'name': 'Lazio', 'code': 'LAZ', 'country': 'Italy', 'founded': 1900}"
                },
                {
                    "role": "assistant",
                    "content": "Lazio, proudly bearing the code LAZ, hails from the heart of Italy. With roots tracing back to 1900, they have a rich history in the panorama of world football, steadily leaving their mark through every passing decade."
                },
                {
                    "role": "user",
                    "content": f"Generate a sentence about the fixture information. The informations are provided in the JSON. {information}"
                },
            ]

        fixture['information'] = self._generate_llama_output(fixture_information_message)
        
        li = []
        for item in information.items():
            for i in str(item[1]).split(' '):
                li.append(i)

        self.fixture_information_parentT.append(self.parent_t_score(fixture['information'], li))

        for team_id in statistics:
            fixture[team_id] = {}
            fixture_statistics_messages = [
                {   
                    "role": "system", 
                    "content": "You are a sports journalist for football (soccer) respond in sentences that can be used in live commentary. Just return the sentences about statistical facts in JSON format. Without introduction what you did."
                },
                {
                    "role": "user",
                    "content": """Give me multiple different sentences about the team statistics. The informations are provided in the JSON. Return the sentences as a python dictionary. For each fact create a key with the topic and the sentecnes as value. Generate text without \ and '\n'. {'team_name': 'Lazio', 'fixtures_played_home': 19, 'fixtures_played_away': 19, 'fixtures_wins_home': 10, 'fixtures_wins_away': 8, 'fixtures_draws_home': 6, 'fixtures_draws_away': 4, 'fixtures_loses_home': 3, 'fixtures_loses_away': 7, 'fixtures_loses_total': 10, 'goals_for_total_home': 42, 'goals_for_total_away': 35, 'goals_for_average_home': '2.2', 'goals_for_average_away': '1.8', 'goals_for_average_total': '2.0', 'goals_for_minute_0-15_total': 12, 'goals_for_minute_0-15_percentage': '16.00%', 'goals_for_minute_16-30_total': 7, 'goals_for_minute_16-30_percentage': '9.33%', 'goals_for_minute_31-45_total': 14, 'goals_for_minute_31-45_percentage': '18.67%', 'goals_for_minute_46-60_total': 13, 'goals_for_minute_46-60_percentage': '17.33%', 'goals_for_minute_61-75_total': 15, 'goals_for_minute_61-75_percentage': '20.00%', 'goals_for_minute_76-90_total': 8, 'goals_for_minute_76-90_percentage': '10.67%', 'goals_for_minute_91-105_total': 6,
                    'goals_for_minute_91-105_percentage': '8.00%', 'goals_for_minute_106-120_total': None, 'goals_for_minute_106-120_percentage': None, 'goals_against_total_home': 25, 'goals_against_total_away': 33, 'goals_against_total_total': 58, 'goals_against_average_home': '1.3', 'goals_against_average_away': '1.7', 'goals_against_average_total': '1.5', 'goals_against_minute_0-15_total': 14, 'goals_against_minute_0-15_percentage': '23.33%', 'goals_against_minute_16-30_total': 8, 'goals_against_minute_16-30_percentage': '13.33%', 'goals_against_minute_31-45_total': 8,
                    'goals_against_minute_31-45_percentage': '13.33%', 'goals_against_minute_46-60_total': 7, 'goals_against_minute_46-60_percentage': '11.67%', 'goals_against_minute_61-75_total': 10, 'goals_against_minute_61-75_percentage': '16.67%', 'goals_against_minute_76-90_total': 7, 'goals_against_minute_76-90_percentage': '11.67%', 'goals_against_minute_91-105_total': 6, 'goals_against_minute_91-105_percentage': '10.00%'}
                            """
                },
                {
                    "role": "assistant",
                    "content": """{
                    \"fixtures_played\": \"Lazio has played an equal amount of games at home and away this season, with a total of 19 each, summing up to a total of 38 fixtures overall.\",
                    \"fixtures_wins\": \"In terms of victories, Lazio has a slight home advantage, with 10 wins at home and 8 on the road, bringing their total fixtures wins to 18.\",
                    \"fixtures_draws_loses\": \"Lazio has had 10 draws and 10 losses in total this season, with a marginally higher draws and losses count at home.\",
                    \"goal_statistics\": \"The team scored a total of 77 goals, with 42 at home and 35 away, which gives them an average of 2.0 goals per game. Regarding conceded goals, they let in 58 goals in total, slightly more away (33) than at home (25).\",
                    \"goal_timing\": \"Lazio tends to score most of their goals between the 61st and 75th minute, while they are most vulnerable to conceding in the opening 15 minutes of the game.\",
                    
                    }"""
                },
                {
                    "role": "user",
                    "content": f"Give me multiple different sentences about the fixture statistics. The informations are provided in the JSON. Return the sentences as a python dictionary. For each fact create a key with the topic and the sentecnes as value. Generate text without \ and '\n'. {statistics[str(team_id)]}"
                },
            ]
            fixture_statistics_output = self._generate_llama_output(fixture_statistics_messages)
            try:
                model_outputs = json.loads(self.correct_json(self.extract_first_match(fixture_statistics_output)))
                output_strings = []
                for statistic in model_outputs:
                    output_strings.append(model_outputs[statistic])

                li = []
                for item in statistics[str(team_id)].items():
                    for i in str(item[1]).split(' '):
                        li.append(i)

                self.fixture_statistics_parentT.append(self.system_level_parent_t_score(output_strings, li))

                fixture[team_id]['statistics'] = model_outputs
            except:
                self.fixture_statistics_parentT.append(0)
                fixture[team_id]['statistics'] = None

        return fixture

    def generate_venue(self):
        venue_messages = [{   
                    "role": "system", 
                    "content": "You are a sports journalist for football (soccer) respond in sentences that can be used in live commentary. Just return the sentence. Without introduction what you did."
                },
                {
                    "role": "user",
                    "content": "Generate a sentence about the team information. The informations are provided in the JSON. {'name': 'Lazio', 'code': 'LAZ', 'country': 'Italy', 'founded': 1900}"
                },
                {
                    "role": "assistant",
                    "content": "Lazio, proudly bearing the code LAZ, hails from the heart of Italy. With roots tracing back to 1900, they have a rich history in the panorama of world football, steadily leaving their mark through every passing decade."
                },
                {
                    "role": "user",
                    "content": f"Generate a sentence about the venue information. The informations are provided in the JSON. {self.get_venue_information()}"
                },
            ]
        model_outputs = self._generate_llama_output(venue_messages)
        li = []
        for item in self.get_venue_information().items():
            for i in str(item[1]).split(' '):
                li.append(i)

        self.venue_information_parentT.append(self.parent_t_score(model_outputs, li))        
        return model_outputs

    def get_team_information(self):
        return self.team_information

    def get_player_information(self):
        return self.player_information
    
    def get_player_statistics(self):
        return self.player_statistics
    
    def get_player_transfers(self):
        return self.player_transfers
    
    def get_player_injuires(self):
        return self.player_injuries
    
    def get_player_news(self):
        player_news_detail = {}
        for player_id in self.player_news:
            player_news_detail[player_id] = {}
            for news in self.player_news[player_id]:
                if news['id'] in self.news_details:
                    player_news_detail[player_id][news['newsHeadline']] = ''.join(value for key, value in sorted(self.news_details[news['id']]['text'].items()))
        player_news_detail = {key: value for key, value in player_news_detail.items() if value}
        return player_news_detail
    
    def get_team_news(self):
        team_news_detail = {}
        for team_id in self.team_news:
            team_news_detail[team_id] = {}
            for news in self.team_news[team_id]:
                if news['id'] in self.news_details:
                    team_news_detail[team_id][news['newsHeadline']] = ''.join(value for key, value in sorted(self.news_details[news['id']]['text'].items()))
        team_news_detail = {key: value for key, value in team_news_detail.items() if value}
        return team_news_detail
    
    def get_team_coach(self):
        for team_id in self.coaches:
            try:
                del self.coaches[team_id]['id']
                del self.coaches[team_id]['name']
                if self.coaches[team_id]['height'] == None:
                    del self.coaches[team_id]['height']
                if self.coaches[team_id]['weight'] == None:
                    del self.coaches[team_id]['weight']
            except:
                pass
        return self.coaches
    
    def get_today_fixture_id(self):
        return self.todays_fixture_id
    
    def get_fixture_information(self):
        fixtures = self.team_fixtures[list(self.team_fixtures.keys())[0]]
        for fixture in fixtures:
            if fixture == self.todays_fixture_id:
                fixture =  self.team_fixtures[str(self.home_team)][fixture]
                try:
                    del fixture['fixture_id']
                    del fixture['league_id']
                    del fixture['league_logo']
                    del fixture['league_name']
                    del fixture['league_country']
                    del fixture['league_flag']
                    del fixture['league_season']
                    del fixture['teams_home_id']
                    del fixture['teams_away_id']
                except:
                    pass

                return fixture
            
    def get_fixture_lineup(self):
        return self.fixture_lineup
    
    def get_player_ids_from_fixture(self, team_id):
        all_ids = set()
        for player_info in self.get_fixture_lineup()[str(team_id)].get('startXI', []):
            player_id = player_info.get('player', {}).get('id')
            if player_id is not None:
                all_ids.add(player_id)

        # Extract player IDs from 'substitutes'
        for substitute_info in self.get_fixture_lineup()[str(team_id)].get('substitutes', []):
            player_id = substitute_info.get('player', {}).get('id')
            if player_id is not None:
                all_ids.add(player_id)

        return list(all_ids)
    
    def get_fixture_statistics(self):
        return self.fixture_statistics

    def get_venue_information(self):
        for venue_id in self.venue_information:
            if self.venue_information[venue_id]['name'] == self.get_fixture_information()['fixture_venue']:
                try:
                    del self.venue_information[venue_id]['team_id']
                except:
                    pass
                return self.venue_information[venue_id]
    
    def _generate_llama_output(self, messages):
        ## point openai library to internal endpoint
        openai.api_key = "1234"
        openai.api_base = "http://10.180.132.23:8182/v1"

        ## specify the used model
        

        model_name = "TheBloke/Llama-2-70B-chat-AWQ"
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=0,
        )
        return response["choices"][0]["message"]["content"]
    
    def extract_first_match(self, input_string):
        pattern = r'\{[^}]*\}'
        matches = re.findall(pattern, input_string)
        
        if matches:
            return matches[0]
        else:
            return input_string

    def correct_json(self, json_string):
        # Remove extra spaces and newlines
        cleaned_json = json_string.replace("\n", "").replace("  ", "")
        
        # Remove double quotes around property names
        cleaned_json = cleaned_json.replace('"{', '{').replace('}"', '}')
        cleaned_json = cleaned_json.replace(',}', '}')
        
        return cleaned_json

    def word_overlap_model(self, g, table_lexical_items):
        return sum(1 for token in g if token in table_lexical_items) / len(g)

    def entailed_precision(self, generated_ngrams, table_lexical_items):
        total_entailment_prob = 0
        total_generated_ngrams = 0
        for g in generated_ngrams:
            total_entailment_prob += self.word_overlap_model(g, table_lexical_items)
            total_generated_ngrams += len(g)
        return total_entailment_prob / total_generated_ngrams if total_generated_ngrams > 0 else 0

    def geometric_average(self, scores):
        return np.exp(np.mean(np.log([score + 1e-10 for score in scores]))) if len(scores) > 0 else 0

    def longest_common_subsequence(self, x, y):
        x = str(x)
        y = str(y)
        seq_matcher = SequenceMatcher(None, x, y)
        match = seq_matcher.find_longest_match(0, len(x), 0, len(y))
        return match.size

    def entailment_recall(self, table_records, generated_text):
        total_recall = sum(self.longest_common_subsequence(record, generated_text) for record in table_records)
        return total_recall / len(table_records) if len(table_records) > 0 else 0

    def parent_t_score(self, generated_text, table_records):
        precision_scores = [self.entailed_precision(ngrams(generated_text.split(), n), set(table_records)) for n in range(1, 6)]
        entailed_precision_score = self.geometric_average(precision_scores)
        recall_score = self.entailment_recall(table_records, generated_text)
        parent_t = (2 * entailed_precision_score * recall_score) / (entailed_precision_score + recall_score) if (entailed_precision_score + recall_score) > 0 else 0
        return parent_t
    
    def system_level_parent_t_score(self, model_outputs, table_records_list):
        total_parent_t_score = sum(self.parent_t_score(generated_text, table_records) for generated_text, table_records in zip(model_outputs, table_records_list))
        return total_parent_t_score / len(model_outputs) if len(model_outputs) > 0 else 0

    def calculate_rouge_scores(self, hypothesis, reference):
        rouge = Rouge()
        scores = rouge.get_scores(hypothesis, reference)
        self.rouge_scores_1.append(scores[0]['rouge-1']['f'])
        self.rouge_scores_2.append(scores[0]['rouge-2']['f'])
        self.rouge_scores_l.append(scores[0]['rouge-l']['f'])
    
    def main(self):
        output_json = {}

        output_json[self.team_ids[0]] = {}
        output_json[self.team_ids[0]]['name'] = self.get_team_name(self.team_ids[0])
        print(f'Generating information for {self.get_team_name(self.team_ids[0])}')
        output_json[self.team_ids[0]]['information'] = self.generate_team_information(self.team_ids[0])
        print(f'Team information parentT score: {self.get_overall_team_information_parentT()}')

        print(f'Generating statistics for {self.get_team_name(self.team_ids[0])}')
        output_json[self.team_ids[0]]['statistics'] = self.generate_team_statistics(self.team_ids[0])
        print(f'Team statistics parentT score: {self.get_overall_team_statistics_parentT()}')

        print(f'Generating news for {self.get_team_name(self.team_ids[0])}')
        output_json[self.team_ids[0]]['news'] = self.generate_team_news(self.team_ids[0])
        print(f'ROUGE score for team news: tbd')

        output_json[self.team_ids[0]]['injuries'] = self.generate_team_injuries(self.team_ids[0])
        print(f'Team injuries parentT score: {self.get_overall_team_injuries_parentT()}')

        output_json[self.team_ids[0]]['players'] = self.generate_team_players(self.team_ids[0])
        print(f'Team players information parentT score: {self.get_overall_player_information_parentT()}')
        print(f'Team players statistics parentT score: {self.get_overall_player_statistics_parentT()}')
        print(f'Team players transfers parentT score: {self.get_overall_player_transfers_parentT()}')
        print(f'ROUGE score for player news: tbd')

        print(f'Generating coach for {self.get_team_name(self.team_ids[0])}')
        output_json[self.team_ids[0]]['coach'] = self.generate_team_coach(self.team_ids[0])
        print(f'Team coach parentT score: {self.get_overall_coach_parentT()}')

        # team 489
        output_json[self.team_ids[1]] = {}
        output_json[self.team_ids[1]]['name'] = self.get_team_name(self.team_ids[1])
        print(f'Generating information for {self.get_team_name(self.team_ids[1])}')
        output_json[self.team_ids[1]]['information'] = self.generate_team_information(self.team_ids[1])
        print(f'Team information parentT score: {self.get_overall_team_information_parentT()}')

        print(f'Generating statistics for {self.get_team_name(self.team_ids[1])}')
        output_json[self.team_ids[1]]['statistics'] = self.generate_team_statistics(self.team_ids[1])
        print(f'Team statistics parentT score: {self.get_overall_team_statistics_parentT()}')

        print(f'Generating news for {self.get_team_name(self.team_ids[1])}')
        output_json[self.team_ids[1]]['news'] = self.generate_team_news(self.team_ids[1])
        print(f'ROUGE score for team news (ROUGE-1, ROUGE-2, ROUGE-L): {self.get_overall_rouge_scores()}')

        output_json[self.team_ids[1]]['injuries'] = self.generate_team_injuries(self.team_ids[1])
        print(f'Team injuries parentT score: {self.get_overall_team_injuries_parentT()}')

        output_json[self.team_ids[1]]['players'] = self.generate_team_players(self.team_ids[1])
        print(f'Team players information parentT score: {self.get_overall_player_information_parentT()}')
        print(f'Team players statistics parentT score: {self.get_overall_player_statistics_parentT()}')
        print(f'Team players transfers parentT score: {self.get_overall_player_transfers_parentT()}')
        print(f'ROUGE score for player news (ROUGE-1, ROUGE-2, ROUGE-L): {self.get_overall_rouge_scores()}')

        print(f'Generating coach for {self.get_team_name(self.team_ids[1])}')
        output_json[self.team_ids[1]]['coach'] = self.generate_team_coach(self.team_ids[1])
        print(f'Team coach parentT score: {self.get_overall_coach_parentT()}')

        # fixture 
        print(f'Generating fixture for {self.get_team_name(self.team_ids[0])} vs {self.get_team_name(self.team_ids[1])}')
        output_json['fixture'] = self.generate_fixture()
        print(f'Team fixture information parentT score: {self.get_overall_fixture_information_parentT()}')
        print(f'Team fixture statistics parentT score: {self.get_overall_fixture_statistics_parentT()}')

        print(f'Generating venue for {self.get_team_name(self.team_ids[0])} vs {self.get_team_name(self.team_ids[1])}')
        output_json['venue'] = self.generate_venue()
        print(f'Team venue parentT score: {self.get_overall_venue_information_parentT()}')

        print(f'save output to llama_output.json')
        with open('llama_output.json', 'w', encoding='utf-8') as outfile:
            json.dump(output_json, outfile, indent=4)

        print(f'save parentT scores to parentT_scores.json')
        with open('parentT_scores.json', 'w') as score_file:
            json.dump({
                'team_information': self.get_overall_team_information_parentT(),
                'team_statistics': self.get_overall_team_statistics_parentT(),
                'team_injuries': self.get_overall_team_injuries_parentT(),
                'player_information': self.get_overall_player_information_parentT(),
                'player_statistics': self.get_overall_player_statistics_parentT(),
                'player_transfers': self.get_overall_player_transfers_parentT(),
                'coach': self.get_overall_coach_parentT(),
                'fixture_information': self.get_overall_fixture_information_parentT(),
                'fixture_statistics': self.get_overall_fixture_statistics_parentT(),
                'venue_information': self.get_overall_venue_information_parentT(),
                'news_rouge_1': self.get_overall_rouge_scores()[0],
                'news_rouge_2': self.get_overall_rouge_scores()[1],
                'news_rouge_l': self.get_overall_rouge_scores()[2]
            }, score_file, indent=4)

        return output_json
    
if __name__ == "__main__":
    generated_output = LlamaGenerate(team_ids=[487, 489]).main()

