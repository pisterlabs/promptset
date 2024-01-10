import os
import sys
import random
from argparse import ArgumentParser
import json
import copy

import openai
import time
import functools

from set_timeline import TimelineSetter
sys.path.append('../')
from type.entities import Entities

def retry_decorator(max_error_count=10, retry_delay=1): # Loop with a maximum of 10 attempts
    def decorator_retry(func):
        functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Initialize an error count
            error_count = 0
            while error_count < max_error_count:
                try:
                    v = func(*args, **kwargs)
                    return v
                except openai.error.Timeout as e:
                    print("Timeout error occurred. Re-running the function.")
                    print(f"Timeout error: {e}")
                    error_count += 1
                except openai.error.APIError as e:
                    print("OPENAI API error occurred. Re-running the function.")
                    print(f"OPENAI API error: {e}")
                    error_count += 1
                except ValueError as e:
                    print("ValueError occurred. Re-running the function.")
                    print(f"ValueError: {e}")
                    error_count += 1
                except TypeError as e: # For when other functions are called in function calling.
                    print("TypeError occurred. Re-running the function.")
                    print(f"TypeError: {e}")
                    error_count += 1
                except openai.error.InvalidRequestError as e:
                    print("InvalidRequestError occurred. Continuing the program.")
                    print(f"openai.error.InvalidRequestError: {e}")
                    break  # Exit the loop
                except Exception as e:
                    print("Exception error occurred. Re-running the function.")
                    print(f"Exeption: {e}")
                    error_count += 1
                time.sleep(retry_delay)  # If an error occurred, wait before retrying
            if error_count == max_error_count:
                sys.exit("Exceeded the maximum number of retries. Exiting the function.")
                return None
        return wrapper
    return decorator_retry

class RougeComparer(TimelineSetter):
    def __init__(self, entities_data: Entities, model_name, temp, min_docs_num_in_1timeline=8, max_docs_num_in_1timeline=10, top_tl=0.5):
        super().__init__(model_name, temp, min_docs_num_in_1timeline, max_docs_num_in_1timeline, top_tl)

        # self.entity_info_list = random.sample(entities_data['data'][0]['entities']['list'], 5)
        self.entity_info_list = entities_data['data'][0]['entities']['list'][236:236+1]

    @retry_decorator(max_error_count=10, retry_delay=1)
    def generate_multiple_timelines(self):
        rouge_1_scores = {
            '1': [],
            '2': [],
            '3': [],
            '4': [],
        }
        rouge_2_scores = {
            '1': [],
            '2': [],
            '3': [],
            '4': [],
        }
        rouge_l_scores = {
            '1': [],
            '2': [],
            '3': [],
            '4': [],
        }

        for i, entity_info in enumerate(self.entity_info_list):
            print('\n')
            print(f"{i+1}/{len(self.entity_info_list)}. ID: {entity_info['ID']}, entity: {entity_info['items']}")
            # Define the number of timelines to generate for this entity
            timeline_num = int(int(entity_info['freq'] / self.max_docs_num_in_1timeline) * self.top_tl)

            # Generate timelines
            # output_data: EntityTimelineData = self.generate_timelines(entity_info, timeline_num)
            # Initialize for analytics list
            self.initialize_list_for_analytics()
            # setter for GPTResponseGetter
            self.set_entity_info(entity_info)

            # prompts
            system_content, user_content_1, user_content_2 = self.get_prompts(entity_info)

            # Loop processing
            for i in range(self.get_max_reexe_num()+1):
                self.set_reexe_num(i)

                # messages
                # messages=[
                #     {'role': 'system', 'content': system_content},
                #     {'role': 'user', 'content': user_content_1}
                # ]
                # messages = self.get_gpt_response_classic(messages, model_name=self.model_name, temp=self.temp)
                # story: str = messages[-1]['content']
                # story = "In the shadow of a disrupted world order, Russia's aggressive stance towards Ukraine has cast a pall over Eastern Europe. Despite Kremlin denials of invasion plans and Senator Kennedy's skepticism on Fox News, intelligence pointed to a sobering truth—the risk of an imminent Russian incursion was alarmingly high. As U.S. and Russian officials faced off in the U.N., Germany halted the Nord Stream 2 pipeline and the world braced for conflict."
                story = "As Russia pulls back forces from Ukraine’s Kharkiv region, tensions remain high. The Biden administration confirms that Russia has purchased rockets from North Korea, while Ukraine reports approximately 9,000 troop losses. Amidst the turmoil, Russia’s intent to annex large portions of Eastern Ukraine has been exposed, Ukraine‘s President Zelenskyy warns that his country may only be the beginning. As Russia targets Ukraine’s industrial heartland, global response ramps up with Joe Biden approving $800 million in military aid. Despite the grim situation, Ukraine stands resilient, refusing to surrender its lands and remaining hopeful that Russia’s strategy will ultimately fail."
                messages = [
                    {'role': 'system', 'content': system_content},
                    {'role': 'user', 'content': user_content_1},
                    {'role': 'assistant', 'content': story}
                ]
                print('Got 1st GPT response.')

                """
                1/4 Only GPT
                2/4 + add by GPT
                3/4 + add by rouge
                """
                messages.append({'role': 'user', 'content': user_content_2})
                timeline_list, IDs_from_gpt = self.get_gpt_response_timeline(messages, model_name=self.model_name, temp=0)
                timeline_list = self.sort_docs_by_date(timeline_list, True)

                #Check
                if self._check_conditions(entity_info, IDs_from_gpt, timeline_list):
                    # If all conditions are met
                    print('1/4 clear.')

                    # Add some docs by GPT
                    self.initialize_timeline_info_archive()
                    entity_info_copied_for2 = copy.deepcopy(entity_info)
                    timeline_list_2, _ = self.generate_timeline_by_chaneg_of_rouge(story, timeline_list, entity_info_copied_for2, useGPT=True)
                    timeline_list_2 = self.sort_docs_by_date(timeline_list_2, True)
                    print('2/4 clear.')

                    # Add some docs by rouge score
                    self.initialize_timeline_info_archive()
                    entity_info_copied_for3 = copy.deepcopy(entity_info)
                    timeline_list_3, _ = self.generate_timeline_by_chaneg_of_rouge(story, timeline_list, entity_info_copied_for3, useGPT=False)
                    timeline_list_3 = self.sort_docs_by_date(timeline_list_3, True)
                    print('3/4 clear.')
                else:
                    print('Failed to get 2nd GPT response (1/3).')
                    continue

                """
                4/4 Only rouge
                """
                # Add some docs by rouge score
                self.initialize_timeline_info_archive()
                entity_info_copied_for4 = copy.deepcopy(entity_info)
                # new_timeline_list, re_generate_flag = self.generate_timeline_by_rouge(story, [], entity_info_copied, useGPT=False)
                new_timeline_list, re_generate_flag = self.generate_timeline_by_chaneg_of_rouge(story, [], entity_info_copied_for4, useGPT=False)

                if not re_generate_flag:
                    timeline_list_4 = self.sort_docs_by_date(new_timeline_list, True)
                    print('4/4 clear.')
                else:
                    print('Missed to create a timeline.')
                    continue

                # All clear!
                for i, tl_list in enumerate([timeline_list, timeline_list_2, timeline_list_3, timeline_list_4]):
                    rouge_scores = self.get_rouge_scores(summary=story, reference=" ".join([f"{doc['headline']} {doc['short_description']}" for doc in tl_list]), alpha=self.rouge_alpha)
                    rouge_1_scores[str(i+1)].append(rouge_scores['rouge_1'])
                    rouge_2_scores[str(i+1)].append(rouge_scores['rouge_2'])
                    rouge_l_scores[str(i+1)].append(rouge_scores['rouge_l'])
                break
            else:
                """
                When not using a timeline that did not exceed the threshold
                """
                sys.exit('NO TIMELINE INFO')

        ave = lambda list: sum(list)/len(list)
        print(
            f"""
            Comparison:\n
            1: rouge_1. {ave(rouge_1_scores['1'])}, rouge_2. {ave(rouge_2_scores['1'])}, rouge_l. {ave(rouge_l_scores['1'])}\n
            2: rouge_1. {ave(rouge_1_scores['2'])}, rouge_2. {ave(rouge_2_scores['2'])}, rouge_l. {ave(rouge_l_scores['2'])}\n
            3: rouge_1. {ave(rouge_1_scores['3'])}, rouge_2. {ave(rouge_2_scores['3'])}, rouge_l. {ave(rouge_l_scores['3'])}\n
            4: rouge_1. {ave(rouge_1_scores['4'])}, rouge_2. {ave(rouge_2_scores['4'])}, rouge_l. {ave(rouge_l_scores['4'])}
            """.replace('            ', '')
            )
        print(f"story: {story}")
        for i, tl_list in enumerate([timeline_list, timeline_list_2, timeline_list_3, timeline_list_4]):
            print(f"\n{i+1}.")
            for doc in tl_list:
                print(f"{doc['date']}\n{doc['headline']}\n{doc['short_description']}")


def main():
    parser = ArgumentParser()
    parser.add_argument('--file_path', default='/mnt/mint/hara/datasets/news_category_dataset/clustering/v1/entities.json')
    parser.add_argument('--out_dir', default='/mnt/mint/hara/datasets/news_category_dataset/clustering/v1/')
    parser.add_argument('--model_name', default='gpt-4')
    parser.add_argument('--temp', default=0.8, type=float, help='Temperature for 1st response of GPT.')
    parser.add_argument('--min_docs', default=4, type=int, help='min_docs_num_in_1timeline')
    parser.add_argument('--max_docs', default=8, type=int, help='max_docs_num_in_1timeline')
    parser.add_argument('--top_tl', default=0.5, type=float, help='top_tl: Number of timelines to be generated, relative to the number of timelines that can be generated.')
    parser.add_argument('--json_file_name', default='no_fake_timelines')
    parser.add_argument('--max_reexe_num', default=2, type=int)
    # For rouge score
    parser.add_argument('--alpha', default=0.8, type=float)
    parser.add_argument('--th_1', default=0.25, type=float)
    parser.add_argument('--th_2', default=0.12, type=float)
    parser.add_argument('--th_l', default=0.15, type=float)
    parser.add_argument('--th_2_rate', default=1.1, type=float)
    parser.add_argument('--th_2_diff', default=0.008, type=float)
    args = parser.parse_args()

    with open(args.file_path, 'r') as F:
        entities_data: Entities = json.load(F)

    rc = RougeComparer(entities_data, args.model_name, args.temp, args.min_docs, args.max_docs, args.top_tl)
    rc.set_max_reexe_num(args.max_reexe_num)
    rc.set_rouge_parms(args.alpha, args.th_1, args.th_2, args.th_l, args.th_2_rate, args.th_2_diff, True)
    rc.generate_multiple_timelines()


if __name__ == '__main__':
    main()