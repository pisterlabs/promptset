from types import SimpleNamespace
from models.question_generation import *
from models.personality_imitation import PersonalityImitation
# from prompts.prompt_template import PersonalityTemplate
from data_processing.preprocess import KaggleDataset, KaggleDatasetWithQuestion
from prompts.prompt_template_new import PersonalityTemplate
from tqdm import tqdm
import csv
import sys, os
import openai
# sys.path.append(os.getcwd())
# args = SimpleNamespace(
#     api_key = 'None',
#     api_base='http://babel-0-19:8769/v1',
#     model='llama2-70b-chat',
#     chat_mode = False,
#     max_tokens = 512,
#     model_kwargs = {"stop": ["</s>", "Questioner:", "Questioner", "<\s>", "</s><s>", "[INSTS]"]},
#     temperature=0,
#     top_p = 1,
# )
# openai.api_key = args.api_key
# openai.api_base = args.api_base
args = SimpleNamespace(
    # api_key = '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',
    api_key = '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',
    # api_base='http://babel-3-19:5050/v1',
    model='gpt-3.5-turbo-0301',
    chat_mode = True,
    max_tokens = 512,
    model_kwargs = {"stop": ["Human", "nHuman", "Human:"]},
    temperature=0.3
)
os.environ['OPENAI_API_KEY']=args.api_key


max_len = 30
max_sentence_len = 400

# question_selected [1, 5, 15]
kaggle_with_question = KaggleDatasetWithQuestion("data/kaggleWithQuestionSelected_3000.csv")
data = kaggle_with_question.poster_data
# 'IE',
# 'JP'
type_list  = ['SN', 'TF']
for t in type_list:
    if t=='IE':
        invoke_chat = 'introversion or extroversion'
    elif t == 'SN':
        invoke_chat = 'intuition or sensing'
    elif t == 'TF':
        invoke_chat = 'thinking or feeling'
    elif t == 'JP':
        invoke_chat = 'judging or perceiving'
    else:
        print("No such predefined type.")
        raise AssertionError
    PItemplate = PersonalityTemplate(invoke_chat, args)
    root_path = 'data/new_gpt3.5/'
    file_path =  root_path + t + '_new_gpt3' + '.csv'
    file = open(file_path, 'a', newline='')
    writer = csv.writer(file, delimiter='\t')
    writer.writerow(['pred', 'label0', 'label1', 'label2', 'label3'])
    file.close()
    for people in tqdm(data):

        count = 0
        select_posts = []
        select_questions = []
        for post, question in zip(people['posts'], people['questions']):
            if count >= max_len or len(post.strip())==0 or question == 'N/A' or len(post.strip())>max_sentence_len or len(question)>max_sentence_len:
                continue
            else:
                select_posts.append('Peter: ' + post)
                select_questions.append('Questioner: ' + question)
                count += 1
        generator = PersonalityImitation(args, system_prompt=PItemplate.system_prompt, questions = select_questions, posts = select_posts, ai_prompt= PItemplate.ai_prompt, user_prompt =PItemplate.user_prompt, json_message= PItemplate.json_message, response_schemas=PItemplate.response_schemas)
        file = open(file_path, 'a', newline='')
        writer = csv.writer(file, delimiter='\t')
        # if 'gpt' in args.model:
        #     type_json = type_json.content
        try:
            type_json = generator.invoke_agent(invoke_chat)
        
            writer.writerow([type_json['type'], people['label0'], people['label1'], people['label2'], people['label3']])
        except:
            writer.writerow(["None", people['label0'], people['label1'], people['label2'], people['label3']])
        file.close()