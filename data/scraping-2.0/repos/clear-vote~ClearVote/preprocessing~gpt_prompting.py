import re
import warnings
import os
import openai
import json
import time
from datetime import datetime

'''
gpt-3.5 turbo (not good enough): $0.18 per run for about 3 minutes, sometimes it just makes shit up
gpt-4: $0.40 for about 9 minutes
TODO: test for bad datasets and implement the ability to manually recorrect
'''
class GPT:
    @staticmethod
    def fetch_rating_for_candidate(candidate_statement, gpt_model, political_value_metric_set, rating_type):
        start_time = time.time()  # Start timer

        political_value_metric_string = ""
        for pvm in political_value_metric_set:
            political_value_metric_string += pvm[0] + ": " + pvm[1] + "\n"
        
        system_prompt = "The following is a list of categories you will judge the user input on:\n\n"
        system_prompt += political_value_metric_string
        system_prompt += '''\n\nThe output should be an array of tuples in JSON format.
        For each tuple, index[0] contains the one-word category as a string, and index[1] should be your rating of how well 
        the user input addresses that category. This rating can be from 0 to 100. 
        Consider both the positive and negative implications of the provided information when evaluating them, 
        with special consideration of their background, education, and past actions.'''

        openai.api_key = os.environ.get("OPENAI_API_KEY")

        while True:
            try:
                response = openai.ChatCompletion.create(
                model=gpt_model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": candidate_statement
                        }
                    ],
                    temperature=0,
                    max_tokens=500,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                break
            except:
                print("ChatGPT needs to cooldown. Waiting 30 seconds...")
                time.sleep(30)
        
        end_time = time.time()  # End timer
        duration_seconds = end_time - start_time
        print(f"      duration seconds: {round(duration_seconds, 2)}")
        
        api_data = json.loads(response['choices'][0]['message']['content'])
        print(api_data)
        return api_data
    
    @staticmethod
    def iterate_contests_and_candidates(candidate_data_set, gpt_model, political_value_metric_set, rating_type):
        result = []
        for contest in sorted(os.listdir(candidate_data_set)):
            if '.DS_Store' in contest:
                print(contest, "contest failed")
                continue
            contest_type = GPT._get_contest_type(contest)
            political_value_metric_set = GPT._get_political_value_metric_set(contest_type)
            print("contest type:", contest_type)
            result.append({"contest_type": contest_type, "districts": []})
            contest_path = os.path.join(candidate_data_set, contest)
            
            for district in sorted(os.listdir(contest_path)):
                if '.DS_Store' in district:
                    print("district failed")
                    continue
                current_contest = len(result) - 1
                position_number = GPT._get_position_number(district)
                print(f"  position number:", position_number)
                result[current_contest]["districts"].append({"position_number": position_number, "candidates": []})
                district_path = os.path.join(contest_path, district)
                
                for candidate in sorted(os.listdir(district_path)):
                    if '.DS_Store' in candidate:  # Only process .json files
                        print("candidate failed")
                        continue
                    
                    current_district = len(result[current_contest]["districts"]) - 1
                    candidate_path = os.path.join(district_path, candidate)
                    with open(candidate_path, 'r') as f:
                        candidate_statement = f.read()
                    print(f"    fetching political value metrics for", candidate)
                    
                    # Sometimes it doesn't work, so we give it 3 attempts
                    attempt = 0
                    while attempt < 3:
                        try:
                            issues = GPT.fetch_rating_for_candidate(candidate_statement, gpt_model, political_value_metric_set, rating_type)
                            if all(item[1] == 0 for item in issues):
                                print(f"        GPT generated a dataset with all zeroes. Generating again")
                                attempt += 1
                                continue
                            scaled_issues = GPT._scale_ratings(issues, 100)
                            candidate_name = os.path.splitext(candidate)[0]
                            result[current_contest]["districts"][current_district]["candidates"].append({"name": candidate_name, "issues": scaled_issues})
                            break
                        except json.JSONDecodeError:
                            print(f"        Failed to decode JSON for", candidate)
                            attempt += 1
                    if attempt == 3:
                        print(f"        Too many attempts. Moving onto the next candidate")
        return result

    @staticmethod
    def _scale_ratings(issues, scalar):
        # Extract values from the nested list format
        total = sum([item[1] for item in issues])
            
        # Scale the values to sum up to the provided scalar (e.g., 100)
        scaled_data = []
        for item in issues:
            key, value = item
            if value == 0:
                scaled_data.append([key, 0])
            else:
                scaled_value = (value / total) * scalar
                scaled_data.append([key, scaled_value])

        return scaled_data

    # Tries to find the contest type from the subdirectory's name
    @staticmethod
    def _get_contest_type(contest):
        if "city" in contest.lower():
            return "city_council"
        elif "port" in contest.lower():
            return "port_commissioner"
        elif "school" in contest.lower():
            return "school_district_director"
        else:
            warnings.warn("Keyword not found, returning 'position_not_found'")
            return "position_not_found"


    # Tries to get the position number from the subdirectories number
    @staticmethod
    def _get_position_number(district):
        numbers = re.findall(r'\d+', district)
        
        # If no numbers are found, return None
        if not numbers:
            warnings.warn("No number found in the input string")
            return None
        
        # Return the last number found in the string
        return int(numbers[-1])
    

    # Given a contest type, this returns the matching political value metric set 
    @staticmethod
    def _get_political_value_metric_set(contest_type):
        with open('preprocessing/political_value_metric_sets/political_value_metrics_1.0.json', 'r') as f:
            pvm_data = json.load(f)
        for pvm_set in pvm_data:
            if pvm_set['contest_type'] == contest_type:
                return pvm_set['issues']
        warnings.warn(f"No issues found for contest_type: {contest_type}")
        return None

    # This calls the political value metric set
    @staticmethod
    def generate_new_pvm_dataset(candidate_data_set, generate_ranking_and_score_files, political_value_metric_set="/preprocessing/poltical_value_metric_sets/political_value_metrics_1.0.json",
                                 municipality="Seattle", state="WA", election_type="Primary", registration_deadline="1690243199", voting_open="1690588800",
                                 voting_close="1690945200", gpt_model="gpt-4"):
        
        scores_json = GPT.iterate_contests_and_candidates(candidate_data_set,
                                                          gpt_model,
                                                          political_value_metric_set,
                                                          "scores")
        scores = { "municipality": municipality, 
                   "state": state,
                   "elections": [
                        { "election_type": election_type,
                          "registration_deadline": registration_deadline,
                          "voting_open": voting_open,
                          "voting_close": voting_close,
                          "contests": scores_json}]}
       
        date_obj = datetime.fromtimestamp(int(voting_open))
        if generate_ranking_and_score_files:
            score_file_name = os.path.join('preprocessing', 'election_datasets', 'wa-seattle-' + date_obj.strftime("%m-%d-%y") + '.json')
            with open(score_file_name, 'w') as f:
                json.dump(scores, f, indent=4)