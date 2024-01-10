from types import SimpleNamespace
from models.question_generation import *
from models.personality_imitation import PersonalityImitation
# from prompts.prompt_template import PersonalityTemplate
from prompts.prompt_template_new import PersonalityTemplate
from langchain.llms.vllm import VLLM
from data_processing.preprocess import KaggleDataset, KaggleDatasetWithQuestion
from tqdm import tqdm
import csv
import sys, os
import openai
sys.path.append(os.getcwd())
args = SimpleNamespace(
    model="/data/datasets/models/hf_cache/models--tiiuae--falcon-7b-instruct/snapshots/cf4b3c42ce2fdfe24f753f0f0d179202fea59c99",
    llm=None,
    chat_mode = True,
    max_tokens = 512,
    model_kwargs = {"stop": ["</s>", "Questioner:", "Questioner", "<\s>", "</s><s>", "[INSTS]"]},
)

if args.model.startswith("/data/"):
    args.llm = VLLM(model=args.model, tensor_parallel_size=1, gpu_memory_utilization=0.95, max_new_tokens=512, top_k=1, model_kwargs=args.model_kwargs)

max_len = 30
max_sentence_len = 400

# question_selected [1, 5, 10, 15]
ab_num = None
kaggle_with_question = KaggleDatasetWithQuestion("data/kaggleWithQuestionSelected_3000.csv")
data = kaggle_with_question.poster_data



type_list  = ['IE', 'NS', 'TF', 'JP']
# type_list  = ['TF', 'JP']
for t in type_list:
    if t=='IE':
        invoke_chat = 'introversion or extroversion'
    elif t == 'NS':
        invoke_chat = 'intuition or sensing'
    elif t == 'TF':
        invoke_chat = 'thinking or feeling'
    elif t == 'JP':
        invoke_chat = 'judging or perceiving'
    else:
        print("No such predefined type.")
        raise AssertionError
    PItemplate = PersonalityTemplate(invoke_chat, args)
    root_path = 'data/Falcon_7B/'
    file_path = root_path + t + '.csv'
    file = open(file_path, 'a', newline='')
    writer = csv.writer(file, delimiter='\t')
    writer.writerow(['pred', 'label0', 'label1', 'label2', 'label3'])
    file.close()
    for people in tqdm(data):

        count = 0
        select_posts = []
        select_questions = []
        for post, question in zip(people['posts'][:ab_num], people['questions'][:ab_num]):
            if count >= max_len or len(post.strip())==0 or question == 'N/A' or len(post.strip())>max_sentence_len or len(question)>max_sentence_len:
                continue
            else:
                select_posts.append('Peter: ' + post)
                select_questions.append('Questioner: ' + question)
                count += 1
        generator = PersonalityImitation(args, system_prompt=PItemplate.system_prompt, questions = select_questions, posts = select_posts, ai_prompt= PItemplate.ai_prompt, user_prompt =PItemplate.user_prompt, json_message=PItemplate.json_message, response_schemas=PItemplate.response_schemas)
        file = open(file_path, 'a', newline='')
        writer = csv.writer(file, delimiter='\t')
        try:
            type_json = generator.invoke_agent(invoke_chat)
            writer.writerow([type_json['type'], people['label0'], people['label1'], people['label2'], people['label3']])
        except:
            writer.writerow(["None", people['label0'], people['label1'], people['label2'], people['label3']])
        file.close()