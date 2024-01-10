import argparse
import openai
import os
import pandas as pd
import sys
import warnings

from utils.profiling import (
    UserProfile,
    Auklets,
    create_userprofiles,
    DEFAULT_DATA
)

from utils.api_messages import (
    get_msg,
    get_msg_with_image
)

from utils.file_interactions import (
    save_result_df,
    read_human_data,
    get_heatmap_descriptions
)

from utils.questionnaire import (
    select_questions,
    find_imagepaths
)

from utils.prompts import (
    SYSTEM,
    USER_PROMPTS,
    TOKENS_LOW
)

from utils.answer_processing import (
    process_llm_output
)

from utils.api_interactions import (
    generate_heatmap_descriptions
)

# openai.api_key = os.environ["OPENAI_API_KEY"]

def initialize_parser():
    """Sets up an argparse.ArgumentParser object with desired command line options."""

    parser = argparse.ArgumentParser(description="Simulate a user study using LLMs.", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--number_users', default=50, type=int, 
                                help="number of users to simulate (default: %(default)s)")
    parser.add_argument('--select_users', default='random', type=str, choices=['random', 'first', 'last'], 
                                help="how to select users (default: %(default)s, choices: %(choices)s)")
    parser.add_argument('--profiling', default='full', type=str, choices=['full', 'minimal', 'none'],
                                help="how much profiling info to use (default: %(default)s, choices: %(choices)s)")
    parser.add_argument('--variation', default=1, type=int, choices=[1, 2, 3, 4],
                                help="what variation of prompting to use (default: %(default)s, choices: %(choices)s)")
    
    subparsers = parser.add_subparsers(dest='subparser_name', help='optionally: specify how to select questionnaire questions. by default, all 20 are used')

    auto_parser = subparsers.add_parser('auto', help='automatic selection of questions. can be given the following args:\n'\
                                        '\t--number_questions\tnumber of questions to select (default: 20)\n'\
                                        '\t--select_questions\thow to select questions (default: \'balanced\', choices: \'random\', \'balanced\', \'first\')')
    auto_parser.add_argument('--number_questions', default=20, type=int,
                                help="number of main questionnaire questions to simulate (default: %(default)s)")
    auto_parser.add_argument('--select_questions', default='balanced', type=str, choices=['random', 'balanced', 'first'],
                                help="how to select the users (default: %(default)s, choices: %(choices)s)")
    
    manual_parser = subparsers.add_parser('manual', help='manual selection of questions. needs following arg:\n'\
                                          '\t--questions\tid(s) of questions to simulate. expects unique values in [1, 20]')
    manual_parser.add_argument('--questions', type=int, nargs='+',
                                help="id(s) of questions to simulate")

    return parser


def LLM_prediction_4(user:UserProfile, image_path: str, profiling_level: str, heatmap_description: str, question: str) -> str:
    """Main part of the interview simulation, variant 4: gives LLM a pre-generated heatmap description, then profiles a user and finally asks a user study question.
    
        Args:
            user (UserProfile) : object representing the user that is simulated
            image_path (str) : path to the image corresponding to the question
            profiling_level (str) : level of profiling to give
            heatmap_description (str) : heatmap description associated with the question
            question (str) : adapted question prompt with necessary profiling info
        
        Returns:
            (str) : simulated question answer
    """
    response = openai.ChatCompletion.create(
            model = "gpt-4-vision-preview",
            max_tokens = 400,
            messages = 
                (get_msg(role="system", prompt=user.profiling_prompt) if profiling_level == 'full' else []) +\
                get_msg_with_image(role="user", prompt=USER_PROMPTS[(4, "intro")]+USER_PROMPTS[(4, "heatmap")], image=image_path) +\
                get_msg(role="assistant", prompt=heatmap_description) +\
                get_msg(role="user", prompt=question) 
        )

    actual_response = response["choices"][0]["message"]["content"] # have a string
    return actual_response


def LLM_prediction_123(user: UserProfile, image_path: str, profiling_level: str, question: str) -> str:
    """Main part of the interview simulation, variants 1/2/3: profiles a user and then asks a user study question.
    
        Args:
            user (UserProfile) : object representing the user that is simulated
            image_path (str) : path to the image corresponding to the question
            profiling_level (str) : level of profiling to give
            question (str) : adapted question prompt with necessary profiling info
        
        Returns:
            (str) : simulated question answer
    """
    response = openai.ChatCompletion.create(
            model = "gpt-4-vision-preview",
            max_tokens = 400,
            messages = 
                (get_msg(role="system", prompt=user.profiling_prompt) if profiling_level == 'full' else []) +\
                get_msg_with_image(role="user", prompt=question, image=image_path)
        )
    actual_response = response["choices"][0]["message"]["content"] # have a string
    return actual_response


def single_prediction(user : UserProfile, image_path : str, q_num : int, profiling_level : str, variation : int, heatmap_description:str=None) -> str:
    """Simulates an interview by first profiling a user and then asking a user study question. 
        Prompts and full response including reasoning are written to "interview_protocol.txt".

        Args:
            user (UserProfile) : object representing the user that is simulated
            image_path (str) : path to the image corresponding to the question
            user_num (int) : number of user in a series of interviews
            q_num (int) : number of question in a series of interviews
            profiling_level (str) : level of profiling to give
            variation (int) : variation of prompts to use
        
        Returns:
            (str) : simulated and cleaned question answer
    """

    # https://platform.openai.com/docs/api-reference/chat/create?lang=python

    if variation == 4:
        QUESTION = ("" if profiling_level == 'none' else user.personalize_prompt(USER_PROMPTS[(4, "profiling")])) + USER_PROMPTS[(4, "question")] + " " + TOKENS_LOW
    else:
        QUESTION = USER_PROMPTS[(variation, "intro")] + ("" if profiling_level == 'none' else user.personalize_prompt(USER_PROMPTS[(variation, "profiling")])) + USER_PROMPTS[(variation, "question")]
    # print(QUESTION)

    # Get gpt-4 response and add the question + answer in the protocol
    with open(PROTOCOL_FILES[variation], mode="a+") as f:
        f.write("Simulated user {u} answering question {i}:\n".format(u=user.user_background['id'], i=q_num))
        if profiling_level == 'full':
            f.write(user.profiling_prompt)
        if variation == 4:
            f.write(USER_PROMPTS[(4, "intro")])
            f.write(USER_PROMPTS[(4, "heatmap")])
            f.write("\n")
            f.write(heatmap_description)
            f.write("\n")
        f.write("\n")
        f.write(QUESTION)
        f.write("\n")

        if variation == 4:
            llm_response = LLM_prediction_4(user, image_path, profiling_level, heatmap_description, QUESTION)
        else:
            llm_response = LLM_prediction_123(user, image_path, profiling_level, QUESTION)
        
        reasoning, answer = process_llm_output(llm_response)
        
        f.write("Reasoning:\n")
        f.write(reasoning)
        f.write("Answer:\n")
        f.write(answer)
        f.write("\n\n")

    return answer


def profile_users(profiles:[UserProfile], profiling_level:str):
    """Personalizes system prompts for users at given profiling level. If profiling level is 'none', system prompt will be None.
    
        Args:
            profiles ([UserProfile]) : the users to profile
            profiling (str) : level of profiling to give
    """
    if profiling_level == 'full':
        for p in profiles:
            p.personalize_prompt(SYSTEM, profiling=True)


def simulate_interviews(question_paths:[(int, str)], profiles:[UserProfile], profiling_level:str, variation:int, heatmap_descriptions:dict[int, str]=None):
    """Simulates interview for each user-question combination and saves results to output file.

        Args:
            question_paths ([(int, str)]) : IDs of questions with associated filepaths for images
            profiles ([UserProfile]) : objects representing the users to simulate
            profiling (str) : level of profiling to give
            variation (int) : prompting variant
            heatmap_descriptions (dict[int, str]) : pre-generated descriptions of the heatmaps (for prompt variation 4)
    """
    # find (previous) results    
    results_df = pd.read_csv(RESULT_FILES[variation], index_col = "id", keep_default_na=False)
    #results_df['LLM_Q2'] = 'NA'

    birds = [bird.value.lower() for bird in Auklets]

    # simulate interview for each user and question
    for user in profiles:

        user_id = user.user_background['id']

        # if the user does not already have a row in the results data frame, create a new one
        if user_id not in list(results_df.index):
            print(user_id)
            results_df.loc[user_id] = 'NA'

        # request gpt-4 responses for not yet (properly) answered questions
        for (q_index, q_path) in question_paths:
            question = "LLM_Q" + str(q_index) # TODO: will have to change this probably
            print(results_df.at[user_id, question])
            if results_df.at[user_id, question].lower() not in birds:
                try:
                    if variation==4:
                        results_df.at[user_id, question] = single_prediction(user, q_path, q_index, profiling_level, variation, heatmap_descriptions[q_index])
                    else:
                        results_df.at[user_id, question] = single_prediction(user, q_path, q_index, profiling_level, variation)
                except Exception as e:
                    # TODO: this does not work
                    print("Response generation failed:\n")
                    print(e)

    # saving the result dataframe again
    save_result_df(results_df)

# openai.api_key = os.environ["OPENAI_API_KEY"]

def main():
    """Sets up and conducts interviews."""

    # parse arguments
    parser = initialize_parser()
    args = parser.parse_args()

    # find questions
    if args.subparser_name is None:
        # if neither manual nor automatic selection of question was chosen, default to all questions
        question_IDs = range(1, 21)
    elif args.subparser_name == 'auto':
        question_IDs = select_questions(args.number_questions, args.select_questions)
    else:
        # question IDs should be unique and between 1 and 20
        question_IDs = set(args.questions)
        valid_IDs = set(range(1, 21))
        if not question_IDs.issubset(valid_IDs):
            warnings.warn("Question IDs outside of valid range [1, 20] will be ignored.")
        question_IDs = list(question_IDs.intersection(valid_IDs))
    print(os.getcwd())
    question_paths = find_imagepaths("prediction_questions.csv", question_IDs)
    
    if args.profiling == 'none':
        profiles:[UserProfile] = [UserProfile(DEFAULT_DATA)]
    else:
        # find users
        profiles:[UserProfile] = create_userprofiles(read_human_data("../../data-exploration-cleanup/cleaned_simulatedusers.csv", 
                                                                  n=args.number_users, selection=args.select_users))
        profile_users(profiles, args.profiling)

    if args.variation != 4:
        simulate_interviews(question_paths, profiles, args.profiling, args.variation)
    else:
        generate_heatmap_descriptions(question_IDs)
        heatmap_descriptions = get_heatmap_descriptions()
        simulate_interviews(question_paths, profiles, args.profiling, args.variation, heatmap_descriptions)


if __name__ == '__main__':
    sys.exit(main())