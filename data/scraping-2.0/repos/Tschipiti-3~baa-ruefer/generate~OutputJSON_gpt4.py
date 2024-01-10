import sys
sys.path.append('/Users/tschip/workspace/baa/baa-ruefer/')
#sys.path.append('/home/jovyan/baa-ruefer/')

from openai import OpenAI
import json
import time
import tqdm
from rouge import Rouge

from GPTGenerate import GPTGenerate
from data_loaders.APIRequests import APIRequests


class OutputJSON():
    def __init__(self, team_ids: list, gpt_api_key=None, requests_api=None):
        self.team_ids = team_ids
        self.gpt_api_key = gpt_api_key
        self.requests_api = requests_api
        self.input_json = GPTGenerate(self.team_ids, self.gpt_api_key, self.requests_api).main()
        #with open('gpt_output.json', 'r') as fp:
            #self.input_json = json.load(fp)

        self.input_json = self.clean_strings(self.input_json)

        self.rouge_scores_1 = []
        self.rouge_scores_2 = []
        self.rouge_scores_l = []

    def clean_strings(self, data):
        if isinstance(data, str):
            try:
                # If the value is a string, perform the cleaning
                return json.loads(data.replace('\n', '').replace('  ', ''))
            except:
                return data
        elif isinstance(data, list):
            # If the value is a list, apply the cleaning to each element in the list
            return [self.clean_strings(item) for item in data]
        elif isinstance(data, dict):
            # If the value is a dictionary, apply the cleaning to each value in the dictionary
            cleaned_dict = {}
            for key, value in data.items():
                cleaned_key = self.clean_strings(key)
                cleaned_value = self.clean_strings(value)
                cleaned_dict[cleaned_key] = cleaned_value
            return cleaned_dict
        else:
            # If the value is not a string, list, or dictionary, return it as is
            return data

    def create_output_json(self, commentary_flavor: str, in_favor=None):
        output_json = {}
        for team_id in list(self.input_json.keys())[:2]:
            in_team_favor = True if str(in_favor) == str(team_id) else False
            not_in_team_favor = True if (in_favor != team_id) and (in_favor != None) else False

            in_favor_text = 'The text should be in favor of the player and team and' if in_team_favor else 'The text should not be in favor of the player and team and' if not_in_team_favor else 'The text should be'

            print(f'Generating output for team {team_id}')
            output_json[team_id] = {}
            output_json[team_id]['team_name'] = self.input_json[team_id]['name']

            print(f'Generating output for team {team_id} information')
            team_information_input = f"Create three outputs with different length. {in_favor_text} with the flavor {commentary_flavor}. Provide the output as a json. The length title is the key. {self.input_json[team_id]['information']}"
            team_information_output = self.generate_gpt_flavor(team_information_input)
            try:
                model_outputs = json.loads(team_information_output)
                for output in model_outputs.values():
                    self.calculate_rouge_scores(output, self.input_json[team_id]['information'])
                output_json[team_id]['information'] = model_outputs
            except:
                print(f'no json format: {team_information_output}')
                self.rouge_scores_1.append(0)
                self.rouge_scores_2.append(0)
                self.rouge_scores_l.append(0)
                output_json[team_id]['information'] = self.format_to_json(team_information_output)
            print(f'Generated output for team {output_json[team_id]["information"]}')

            print(f'Generating output for team {team_id} statistics')
            output_json[team_id]['statistics'] = {}
            for statistic in tqdm.tqdm(self.input_json[team_id]['statistics']):
                print(f'Generating output for team {team_id} statistics {statistic}')
                team_statistics_input = f"Create three outputs with different length. {in_favor_text} with the flavor {commentary_flavor}. Provide the output as a json. The length title is the key. {self.input_json[team_id]['statistics'][statistic]}"
                team_statistics_output = self.generate_gpt_flavor(team_statistics_input)
                try:
                    model_outputs = json.loads(team_statistics_output)
                    for output in model_outputs.values():
                        self.calculate_rouge_scores(output, self.input_json[team_id]['statistics'][statistic])
                    output_json[team_id]['statistics'][statistic] = model_outputs
                except:
                    print(f'no json format: {team_statistics_output}')
                    self.rouge_scores_1.append(0)
                    self.rouge_scores_2.append(0)
                    self.rouge_scores_l.append(0)
                    output_json[team_id]['statistics'][statistic] = self.format_to_json(team_statistics_output)

            if self.input_json[team_id]['news'] != None:
                print(f'Generating output for team {team_id} news')
                output_json[team_id]['news'] = {}
                for news in tqdm.tqdm(self.input_json[team_id]['news']):
                    team_news_input = f"Create three outputs with different length. {in_favor_text} with the flavor {commentary_flavor}. Provide the output as a json. The length title is the key. {self.input_json[team_id]['news'][news]}"
                    team_news_output = self.generate_gpt_flavor(team_news_input)
                    try:
                        model_outputs = json.loads(team_news_output)
                        for output in model_outputs.values():
                            self.calculate_rouge_scores(output, self.input_json[team_id]['news'][news])
                        output_json[team_id]['news'][news] = model_outputs
                    except:
                        print(f'no json format: {team_news_output}')
                        self.rouge_scores_1.append(0)
                        self.rouge_scores_2.append(0)
                        self.rouge_scores_l.append(0)
                        
                        output_json[team_id]['news'][news] = self.format_to_json(team_news_output)

            if self.input_json[team_id]['injuries'] != {}:
                output_json[team_id]['injuries'] = {}
                print(f'Generating output for team {team_id} injuries')
                for player_id in tqdm.tqdm(self.input_json[team_id]['injuries']):
                    print(f'Generating output for player {player_id}')
                    team_injuries_input = f"Create three outputs with different length. {in_favor_text} with the flavor {commentary_flavor}. Provide the output as a json. The length title is the key. {self.input_json[team_id]['injuries'][player_id]}"
                    team_injuries_output = self.generate_gpt_flavor(team_injuries_input)
                    try:
                        model_outputs = json.loads(team_injuries_output)
                        for output in model_outputs.values():
                            self.calculate_rouge_scores(output, self.input_json[team_id]['injuries'][player_id])
                        output_json[team_id]['injuries'][player_id] = model_outputs
                    except:
                        print(f'no json format: {team_injuries_output}')
                        self.rouge_scores_1.append(0)
                        self.rouge_scores_2.append(0)
                        self.rouge_scores_l.append(0)
                        
                        output_json[team_id]['injuries'][player_id] = self.format_to_json(team_injuries_output)

            print(f'Generating output for team {team_id} players')
            output_json[team_id]['players'] = {}
            counter = 1
            n_players = len(self.input_json[team_id]['players'])
            for player_id in tqdm.tqdm(self.input_json[team_id]['players']):
                print(f'Generating output for player {player_id}')
                print(f'Player {counter} of {n_players}')
                output_json[team_id]['players'][player_id] = {}

                print(f'Generating output for player {player_id} information')
                player_information_input = f"Create three outputs with different length. {in_favor_text} with the flavor {commentary_flavor}. Provide the output as a json. The length title is the key. {self.input_json[team_id]['players'][player_id]['information']}"
                player_information_output = self.generate_gpt_flavor(player_information_input)
                try:
                    model_outputs = json.loads(player_information_output)
                    for output in model_outputs.values():
                        self.calculate_rouge_scores(output, self.input_json[team_id]['players'][player_id]['information'])
                    output_json[team_id]['players'][player_id]['information'] = model_outputs
                except:
                    print(f'no json format: {player_information_output}')
                    self.rouge_scores_1.append(0)
                    self.rouge_scores_2.append(0)
                    self.rouge_scores_l.append(0)
                    output_json[team_id]['players'][player_id]['information'] = self.format_to_json(player_information_output)

                print(f'Generating output for player {player_id} statistics')
                output_json[team_id]['players'][player_id]['statistics'] = {}
                if type(self.input_json[team_id]['players'][player_id]['statistics']) != str:
                    for statistic in self.input_json[team_id]['players'][player_id]['statistics']:
                        print(f'Generating output for player {player_id} statistics {statistic}')
                        player_statistics_input = f"Create three outputs with different length. {in_favor_text} with the flavor {commentary_flavor}. Provide the output as a json. The length title is the key. {self.input_json[team_id]['players'][player_id]['statistics'][statistic]}"
                        player_statistics_output = self.generate_gpt_flavor(player_statistics_input)
                        try:
                            model_outputs = json.loads(player_statistics_output)
                            for output in model_outputs.values():
                                self.calculate_rouge_scores(output, self.input_json[team_id]['players'][player_id]['statistics'][statistic])
                            output_json[team_id]['players'][player_id]['statistics'][statistic] = model_outputs
                        except:
                            print(f'no json format: {player_statistics_output}')
                            self.rouge_scores_1.append(0)
                            self.rouge_scores_2.append(0)
                            self.rouge_scores_l.append(0)
                            output_json[team_id]['players'][player_id]['statistics'][statistic] = self.format_to_json(player_statistics_output)
                else:
                    player_statistics_input = f"Create three outputs with different length. {in_favor_text} with the flavor {commentary_flavor}. Provide the output as a json. The length title is the key. {self.input_json[team_id]['players'][player_id]['statistics']}"
                    output_json[team_id]['players'][player_id]['statistics'] = self.generate_gpt_flavor(player_statistics_input)

                
                time.sleep(1)

                print(f'Generating output for player {player_id} transfers')
                output_json[team_id]['players'][player_id]['transfers'] = {}
                if type(self.input_json[team_id]['players'][player_id]['transfers']) != str:
                    for transfer in tqdm.tqdm(self.input_json[team_id]['players'][player_id]['transfers']):
                        print(f'Generating output for player {player_id} transfers {transfer}')
                        player_transfers_input = f"Create three outputs with different length. {in_favor_text} with the flavor {commentary_flavor}. Provide the output as a json. The length title is the key. {self.input_json[team_id]['players'][player_id]['transfers'][transfer]}"
                        player_transfers_output = self.generate_gpt_flavor(player_transfers_input)
                        try:
                            model_outputs = json.loads(player_transfers_output)
                            for output in model_outputs.values():
                                self.calculate_rouge_scores(output, self.input_json[team_id]['players'][player_id]['transfers'][transfer])
                            output_json[team_id]['players'][player_id]['transfers'][transfer] = model_outputs
                        except:
                            print(f'no json format: {player_transfers_output}')
                            self.rouge_scores_1.append(0)
                            self.rouge_scores_2.append(0)
                            self.rouge_scores_l.append(0)

                            output_json[team_id]['players'][player_id]['transfers'][transfer] = self.format_to_json(player_transfers_output)
                else:
                    player_transfers_input = f"Create three outputs with different length. {in_favor_text} with the flavor {commentary_flavor}. Provide the output as a json. The length title is the key. {self.input_json[team_id]['players'][player_id]['transfers']}"
                    output_json[team_id]['players'][player_id]['transfers'] = self.generate_gpt_flavor(player_transfers_input)

                time.sleep(1)

                if self.input_json[team_id]['players'][player_id]['news'] != None:
                    print(f'Generating output for player {player_id} news')
                    output_json[team_id]['players'][player_id]['news'] = {}
                    if type(self.input_json[team_id]['players'][player_id]['news']) != str:
                        for news in tqdm.tqdm(self.input_json[team_id]['players'][player_id]['news']):
                            print(f'Generating output for player {player_id} news {news}')
                            player_news_input = f"Create three outputs with different length. {in_favor_text} with the flavor {commentary_flavor}. Provide the output as a json. The length title is the key. {self.input_json[team_id]['players'][player_id]['news'][news]}"
                            output_json[team_id]['players'][player_id]['news'][news] = self.generate_gpt_flavor(player_news_input)
                    else:
                        player_news_input = f"Create three outputs with different length. {in_favor_text} with the flavor {commentary_flavor}. Provide the output as a json. The length title is the key.{self.input_json[team_id]['players'][player_id]['news']}"
                        player_news_output = self.generate_gpt_flavor(player_news_input)
                        try:
                            model_outputs = json.loads(player_news_output)
                            for output in model_outputs.values():
                                self.calculate_rouge_scores(output, self.input_json[team_id]['players'][player_id]['news'][news])
                            output_json[team_id]['players'][player_id]['news'] = model_outputs
                        except:
                            print(f'no json format: {player_news_output}')
                            self.rouge_scores_1.append(0)
                            self.rouge_scores_2.append(0)
                            self.rouge_scores_l.append(0)

                            output_json[team_id]['players'][player_id]['news'] = self.format_to_json(player_news_output)
                
                counter += 1
                with open('json_output_dump.json', 'w', encoding='UTF-8') as fp:
                    json.dump(output_json, fp, indent=4, ensure_ascii=False)

            print(f'Generating output for team {team_id} coach')
            output_json[team_id]['coach'] = {}
            if type(self.input_json[team_id]['coach']) != str:
                for coach_information in tqdm.tqdm(self.input_json[team_id]['coach']):
                    print(f'Generating output for team {team_id} coach {coach_information}')
                    player_coaches_input = f"Create three outputs with different length. {in_favor_text} with the flavor {commentary_flavor}. Provide the output as a json. The length title is the key. {self.input_json[team_id]['coach'][coach_information]}"
                    player_coaches_output = self.generate_gpt_flavor(player_coaches_input)
                    try:
                        model_outputs = json.loads(player_coaches_output)
                        for output in model_outputs.values():
                            self.calculate_rouge_scores(output, self.input_json[team_id]['coach'][coach_information])
                        output_json[team_id]['coach'][coach_information] = model_outputs
                    except:
                        print(f'no json format: {player_coaches_output}')
                        self.rouge_scores_1.append(0)
                        self.rouge_scores_2.append(0)
                        self.rouge_scores_l.append(0)

                        output_json[team_id]['coach'][coach_information] = self.format_to_json(player_coaches_output)
            else:
                player_coaches_input = f"Create three outputs with different length. {in_favor_text} with the flavor {commentary_flavor}. Provide the output as a json. The length title is the key.{self.input_json[team_id]['coach']}"
                output_json[team_id]['coach'] = self.generate_gpt_flavor(player_coaches_input)
        
        print(f'Generating output for fixture')
        output_json['fixture'] = {}
        for id in self.input_json['fixture']:
            output_json['fixture'][id] = {}
            print(f'Generating output for {id}')
            if id == 'information':
                fixture_information_input = f"Create three outputs with different length. {in_favor_text} with the flavor {commentary_flavor}. Provide the output as a json. The length title is the key. {self.input_json['fixture']['information']}"
                fixture_information_output = self.generate_gpt_flavor(fixture_information_input)
                try:
                    model_outputs = json.loads(fixture_information_output)
                    for output in model_outputs.values():
                        self.calculate_rouge_scores(output, self.input_json['fixture']['information'])
                    output_json['fixture'][id] = model_outputs
                except:
                    print(f'no json format: {fixture_information_output}')
                    self.rouge_scores_1.append(0)
                    self.rouge_scores_2.append(0)
                    self.rouge_scores_l.append(0)

                    output_json['fixture'][id] = self.format_to_json(fixture_information_output)
            else:
                output_json['fixture'][id]['statistics'] = {}
                if type(self.input_json['fixture'][id]['statistics']) != str:
                    for statistic in tqdm.tqdm(self.input_json['fixture'][id]['statistics']):
                        print(f'Generating output for {id} {statistic}')
                        fixture_statistics_input = f"Create three outputs with different length. {in_favor_text} with the flavor {commentary_flavor}. Provide the output as a json. The length title is the key.{self.input_json['fixture'][id]['statistics'][statistic]}"
                        fixture_statistics_output = self.generate_gpt_flavor(fixture_statistics_input)
                        try:
                            model_outputs = json.loads(fixture_statistics_output)
                            for output in model_outputs.values():
                                self.calculate_rouge_scores(output, self.input_json['fixture'][id]['statistics'][statistic])
                            output_json['fixture'][id]['statistics'][statistic] = model_outputs
                        except:
                            print(f'no json format: {fixture_statistics_output}')
                            self.rouge_scores_1.append(0)
                            self.rouge_scores_2.append(0)
                            self.rouge_scores_l.append(0)

                            output_json['fixture'][id]['statistics'][statistic] = self.format_to_json(fixture_statistics_output)
                else:
                    fixture_statistics_input = f"Create three outputs with different length. {in_favor_text} with the flavor {commentary_flavor}. Provide the output as a json. The length title is the key.{self.input_json['fixture'][id]['statistics']}"
                    output_json['fixture'][id]['statistics'] = self.generate_gpt_flavor(fixture_statistics_input)

        print(f'Generating output for venue')
        venue_information_input = f"Create three outputs with different length. {in_favor_text} with the flavor {commentary_flavor}. Provide the output as a json. The length title is the key.{self.input_json['venue']}"
        output_json['venue'] = self.generate_gpt_flavor(venue_information_input)

        return output_json

    def generate_gpt_flavor(self, input):
        client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=self.gpt_api_key,
        )

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You get a text from a sports journalist for live commentary. Take the text and create three outputs with different length. A short in one sentence. A middle one with at most 2 sentences. And a long with maximum 3 sentences. Write the text with the provided flavor."},
                {"role": "user", "content": f"{input}"},
            ],
            model="gpt-4",
            temperature=0.9,
        )
        return response.choices[0].message.content
    
    def format_to_json(self, input):
        client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=self.gpt_api_key,
        )

        response = client.chat.completions.create(
             messages = [
                {"role": "system", "content": "The output is not in json format. Please format it to json."},
                {"role": "user", "content": f"Format string to JSON string: {input}"},
            ],
            model="gpt-4",
            temperature=0.9,
        )

        print(f'Response: {response.choices[0].message.content}')
        return response.choices[0].message.content
    
    def calculate_rouge_scores(self, hypothesis, reference):
        rouge = Rouge()
        scores = rouge.get_scores(hypothesis, reference)
        self.rouge_scores_1.append(scores[0]['rouge-1']['f'])
        self.rouge_scores_2.append(scores[0]['rouge-2']['f'])
        self.rouge_scores_l.append(scores[0]['rouge-l']['f'])

    def get_overall_rouge_scores(self):
        return (sum(self.rouge_scores_1) / len(self.rouge_scores_1)) if len(self.rouge_scores_1) > 0 else None, (sum(self.rouge_scores_2) / len(self.rouge_scores_2)) if len(self.rouge_scores_2) > 0 else None, (sum(self.rouge_scores_l) / len(self.rouge_scores_l)) if len(self.rouge_scores_l) > 0 else None
    
    

if __name__ == "__main__":
    team_ids = [487, 489]
    #request_api = APIRequests(api_key=API-KEY, team_ids=team_ids, season=2022, league_id=135, fixture_date="21-10-2022", max_requests=99)
    output_json = OutputJSON(team_ids, API-KEY, requests_api=None)
    flavor = 'educational'
    in_favor = 487
    output_file = output_json.create_output_json(flavor, in_favor=in_favor)

    with open(f'gpt_final_output_{flavor}.json', 'w', encoding='UTF-8') as fp:
        json.dump(output_file, fp, indent=4, ensure_ascii=False)

    with open(f'gpt_rouge_score.json', 'w', encoding='UTF-8') as f:
        json.dump({'rouge-1': output_json.get_overall_rouge_scores()[0], 'rouge-2': output_json.get_overall_rouge_scores()[1], 'rouge-l': output_json.get_overall_rouge_scores()[2]}, f, indent=4, ensure_ascii=False)

            