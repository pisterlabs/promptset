import openai
from my_secrets import my_secrets
openai_key = my_secrets.get('openai_key')
openai.api_key = openai_key

import random
random.seed(37)
import json
from blocks.models import Question, Answer, Rubric, RubricList

from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
modelPath = 'all-MiniLM-L6-v2'
model = SentenceTransformer(modelPath)
from sklearn.cluster import AgglomerativeClustering

# e.g. "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hello!"}]
def prompt_chatgpt(prompt):
    model="gpt-3.5-turbo-16k"
    # model="gpt-3.5-turbo"
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=prompt,
            temperature=0.0,
        )
        if "error" in response:
            print("OPENAI ERROR: {}".format(response))
            return "ERROR"
        else:
            return response["choices"][0]["message"]["content"]
    except Exception as e: 
        print(e)
        return "ERROR"

def prompt_gpt4(prompt):
    model="gpt-4"
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=prompt,
            temperature=0.0,
        )
        if "error" in response:
            print("OPENAI ERROR: {}".format(response))
            return "ERROR"
        else:
            return response["choices"][0]["message"]["content"]
    except Exception as e: 
        print(e)
        return "ERROR"

def create_rubrics(question, answers, num_samples=40):
    random_answers = random.sample(list(answers), num_samples)
    max_grade = 2

    answers_str = "\n".join(["{}. {}".format(answer.id, answer.answer_text) for i, answer in enumerate(random_answers)])
    # system_prompt = f"""You are an expert instructor for your given course. You've given the short-answer, open-ended question "{question.question_text}" on a recent final exam. Given the following random assortment of students' answers, please come up with a set of 10 rubrics that are of high quality: mutually exclusive, easily understood, reflective, and encompasses all kinds of answers (even the unseen). Rubrics can be positive or negative (covering all potential misunderstandings of the concept presented) and multiple rubrics can apply to the same answers. The maximum amount of points/grade for this answer is {max_grade}, please assign points accordingly (and include rubrics with zeroes, showing what it means to be incorrect). \n\n{answers_str}"""
    system_prompt = f"""You are an expert instructor for your given course. You've given the short-answer, open-ended question "{question.question_text}" on a recent final exam. The following is a random assortment of answers to the question by your students (formatted: <answer ID>. <answer>): \n\n{answers_str}"""
    # user_prompt = """For each rubric generated, please STRICTLY follow the JSON format: {"rubric": "<high-level concept for this rubric>", "answers": "<comma-separated list of answer numbers that fit this rubric>", "points": "<points - from the max to 0>", "explanation": "<elaboration on what really counts for this rubric>"} """
    user_prompt = """Please come up with a set of 5 rubrics that are of high quality: mutually exclusive, easily understood, reflective, and encompasses all kinds of answers (even the unseen). For the output, create a comma-separated list of python dictionaries that STRICTLY follow the JSON format: [{"rubric": "<high-level concept for this rubric>", "answer_ids": "<comma-separated list IDs of top 5 representative answers that fit this rubric>", "explanation": "<elaboration on what really counts for this rubric>"}, ...] """

    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    response = prompt_chatgpt(msgs)

    try:
        rubrics = json.loads(response)
    except Exception as e:
        print(e)
        rubrics = []

    return rubrics, msgs

def tag_answers(question, answers, rubrics, num_samples=40):
    random_answers = random.sample(list(answers), num_samples)

    answers_str = "\n".join(["{}. {}".format(answer.id, answer.answer_text) for i, answer in enumerate(random_answers)])
    rubrics_str = "\n".join(["R{}. {} (meaning: {})".format(i+1, rubric["rubric"], rubric['explanation']) for i, rubric in enumerate(rubrics)])
    system_prompt = f"""You are an expert instructor for your given course. You've given the short-answer, open-ended question "{question.question_text}" on a recent final exam. You and your expert instructor partner created the following rubrics for this question (labelled R<rubric number> below): \n\n{rubrics_str}"""
    user_prompt = """Your task is to assign the rubric labels to each of the following students' answers (formatted: <answer ID>. <answer>). Please assign labels to each of the answers provided. Each answer can have multiple rubrics applied too. Treat this as a multi-class classification task. Please provide reasoning for your labels as well. For the output, create a comma-separated list of python dictionaries that STRICTLY follow the JSON format: [{"answer_id": "<id of the answer>", "rubrics": "<comma-separated list of rubrics (labelled R<number>) that apply to this answer>", "reasoning": "<reason you think the rubrics you chose apply to this answer>"}, ...]\n\nStudents' Answers:\n""" + answers_str

    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    response = prompt_chatgpt(msgs)

    try:
        tags = json.loads(response)
    except Exception as e:
        print(e)
        tags = []

    return tags, msgs

def cluster_answers(df, n_clusters=20):
    sentences = df["answer_text"].apply(str).tolist()

    # Compute SBERT embeddings
    model = SentenceTransformer(modelPath)
    embeddings = model.encode(sentences, convert_to_tensor=True)
    # cosine_scores = util.cos_sim(embeddings, embeddings)

    # Cluster sentences with AgglomerativeClustering
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters) #, affinity='cosine', linkage='average', distance_threshold=threshold)
    clustering_model.fit(embeddings)

    # Assign the cluster labels to the dataframe
    df['cluster'] = clustering_model.labels_
    return df

def create_rubric_suggestions(question, answers, rubrics, polarity="positive", using_df=False):

    if (using_df):
        cluster_df = cluster_answers(answers, n_clusters=20)
        samples = []
        num_clusters = cluster_df['cluster'].nunique()
        for cluster_id in range(num_clusters): 
            sample = cluster_df[cluster_df['cluster'] == cluster_id].sample(n=1)
            for index, row in sample.iterrows():
                # convert sample to dict and add to samples
                sample_dict = row.to_dict()
                samples.append(sample_dict)
        answers = samples
        answers_str = "\n".join(["{}. {}".format(answer['id'], answer['answer_text']) for i, answer in enumerate(answers)])
    else: 
        answers_str = "\n".join(["{}. {}".format(answer.id, answer.answer_text) for i, answer in enumerate(answers)])

    rubrics_str = "\n\n".join(["- {} (polarity: {}, description: {})".format(rubric["title"], rubric["polarity"], rubric['description']) for i, rubric in enumerate(rubrics) if rubric['id'] != 0])
    system_prompt = f"""You are an expert instructor for your given course. You've given the short-answer, open-ended question "{question.question_text}" on a recent final exam. You previously created the following rubrics for this question: \n\n{rubrics_str}"""

    user_prompt = """
    Given the rubrics mentioned and the following student answers (formatted: <answer ID>. <answer>):\n""" + answers_str + """

    Create a set of 2 additional """ + polarity + """ rubrics that are of high quality: mutually exclusive, easily understood, reflective, and encompasses all kinds of answers (even the unseen). For the output, create a comma-separated list of python dictionaries that STRICTLY follow the JSON format:

    [{"id": <always 0 because that's how we identify suggestions>, "polarity": """ + polarity + """, "title": "<short title of the rubric (2-7 words)>", "description": "<longer description of the rubric>", "reasoning_dict": {}}, ...]
    """

    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    response = prompt_chatgpt(msgs)

    try:
        suggestions = json.loads(response)
        for suggestion in suggestions:
            suggestion["id"] = 0
    except Exception as e:
        print(e)
        suggestions = []

    return suggestions

def create_themes(df, x, question_text, num_samples_per_cluster=5):
    cluster_ids = sorted(df['cluster'].unique())
    system_prompt = f"You are an expert instructor for your given course. Currently, you are evaluating student responses to the question: '{question_text}' from a recent final exam. By examining diverse clusters, we hope to inspire more detailed insights based on the variations observed. Based on the provided answers, identify the main themes or recurrent topics that students emphasized.\n\nCome up with a list of themes that effectively encapsulate the types of answers in the dataset. Please output in the following format (one for each cluster): - <theme title>: <theme description kept to 15 words maximum>"
    responses = []

    # iterate through the clusters in batches of x
    for i in range(0, len(cluster_ids), x):
        selected_clusters = cluster_ids[i:i+x]
        user_prompt = ""
        for cluster_id in selected_clusters:
            sample = df[df['cluster'] == cluster_id].sample(n=num_samples_per_cluster, replace=True)
            cluster_string = f"\n\nCluster {cluster_id}: \n\n" + "\n\n".join(sample['answer_text'].tolist())
            user_prompt += cluster_string

        msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = prompt_gpt4(msgs)
        responses.append(response)

    return responses

def create_rubric_suggestions_2(df, answers, question_text):
    # GPT generates themes
    responses = create_themes(df, x=2, question_text=question_text, num_samples_per_cluster=5)
    summarizing_prompt = f"You will be provided a list of themes intended for evaluating student responses on the question: '{question_text}'. Please select ONLY 10 line items to keep and eliminate those that are either too similar to points you select or not directly relevant to the core topic."
    all_themes = "\n\n".join(responses)
    msgs = [{"role": "system", "content": summarizing_prompt}, {"role": "user", "content": all_themes}]
    themes_response = prompt_gpt4(msgs)

    # GPT generates rubrics
    rubrics_prompt = f"Based on the provided answers below, generate 5 positive (common good answers) and 5 negative (potential misunderstandings) rubric items based on the given answers and the themes generated. Please output in the following format: \n- <rubric title>: <rubric description kept to 15 words maximum> (example: <example answer from the dataset provided>) \n\n{answers}"
    msgs.append({"role": "assistant", "content": themes_response})
    msgs.append({"role": "user", "content": rubrics_prompt})
    rubrics_response = prompt_gpt4(msgs)

    formatting_prompt = """For the output, create a comma-separated list of python dictionaries that STRICTLY follow the JSON format:

    [{"id": <unique ID for each rubric, starting from 1>, "polarity": "<positive/negative>", "title": "<short title of the rubric (2-7 words)>", "description": "<longer description of the rubric>", "examples": ["<first example answer from dataset>", "<second example answer from dataset>"], "reasoning_dict": {}}, ...]
    """
    msgs.append({"role": "assistant", "content": rubrics_response})
    msgs.append({"role": "user", "content": formatting_prompt})
    formatted_response = prompt_gpt4(msgs)

    try:
        suggestions = json.loads(formatted_response)
    except Exception as e:
        print(e)
        suggestions = []

    return suggestions

def apply_rubrics(question, answers, rubrics, existing_tags=None):

    answers = sorted(answers, key=lambda x: x.id)

    def convert_reasoning_dict(reasoning_dict):
        # convert dict of answer ID keys and reasoning values to a new-line separated string of Answer texts (from DB) and the reasoning
        reasoning_str = ""
        for answer_id, reasoning in reasoning_dict.items():
            reasoning_str += "{} {}\n".format(Answer.objects.get(id=answer_id).answer_text, f"(REASON: {reasoning})" if reasoning != "" else "")
        return reasoning_str if reasoning_str != "" else "None"

    answers_str = "\n".join(["{}. {}".format(answer.id, answer.answer_text) for i, answer in enumerate(answers)])
    rubrics_str = "\n\n".join(["R{}. {} (polarity: {}, meaning: {})\nR{} Examples:\n{}".format(rubric["id"], rubric["title"], rubric["polarity"], rubric['description'], rubric["id"], convert_reasoning_dict(rubric["reasoning_dict"])) for i, rubric in enumerate(rubrics) if rubric['id'] != 0])
    system_prompt = f"""You are an expert instructor for your given course. You've given the short-answer, open-ended question "{question.question_text}" on a recent final exam. You and your expert instructor partner created the following rubrics for this question (labelled R<rubric number> below, along with examples that your partner annotated with reasoning): \n\n{rubrics_str}"""
    

    user_prompt = """
    Given the rubrics mentioned and the following student answers (formatted: <answer ID>. <answer>):\n""" + answers_str + """

    Label and highlight each student's answer based on the rubric(s) that applies to it. Each answer can have multiple rubrics applied, so treat this as a multi-label classification task. Only highlight the most relevant words per rubric that you choose to apply - keep it short! Please provide reasoning for your labels and a relevancy score as well. For the output, create python dictionary that STRICTLY follow the JSON format:

    {"answer_id": [
        {
            "rubric": "<rubric that applies to this answer (labelled R<number>)>",
            "reasoning": "<reason why rubric R<number> applies>",
            "highlighted": "<substring within the answer of 3-6 words that best supports the reasoning>",
            "relevancy": "<0.5 or 1 to indicate partial or full relevance to the answer>"
        },
        {
            "rubric": "<rubric that applies to this answer (labelled R<number>)>",
            "reasoning": "<reason why rubric R<number> applies>",
            "highlighted": "<substring within the answer of 3-6 words that best supports the reasoning>",
            "relevancy": "<0.5 or 1 to indicate partial or full relevance to the answer>"
        },
        ...
    ], ...}
    """

    # Create a string of existing tags in the format <answer text> [R<rubric number> | <relevancy score> | <reasoning>]
    existing_tags_str = ""
    if (existing_tags is not None):
        # get only tags with relevancy > 0.0
        filtered_tags = [tag for tag in existing_tags.order_by('?') if float(tag.get_reasoning_dict()['relevancy']) > 0.0]
        for tag in filtered_tags[:10]:
            curr_ans = Answer.objects.get(id=tag.answer_id)
            curr_reasoning_dict = tag.get_reasoning_dict()
            if(float(curr_reasoning_dict['relevancy']) > 0.0): existing_tags_str += f"- {curr_ans.answer_text} [Rubric: {tag.tag} | Relevance: {curr_reasoning_dict['relevancy']} | Reason: {curr_reasoning_dict['reasoning']}]\n"

        user_prompt += "\n\n" + "Examples:\n" + existing_tags_str

    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    response = prompt_gpt4(msgs)

    try:
        tags = json.loads(response)
    except Exception as e:
        print(e)
        tags = []

    return tags

def apply_rubrics_2(question, answers, rubrics, existing_tags=None):

    answers = sorted(answers, key=lambda x: x.id)

    def convert_reasoning_dict(reasoning_dict):
        # convert dict of answer ID keys and reasoning values to a new-line separated string of Answer texts (from DB) and the reasoning
        reasoning_str = ""
        for answer_id, reasoning in reasoning_dict.items():
            reasoning_str += "{} {}\n".format(Answer.objects.get(id=answer_id).answer_text, f"(REASON: {reasoning})" if reasoning != "" else "")
        return reasoning_str if reasoning_str != "" else "None"

    answers_str = "\n".join(["{}. {}".format(answer.id, answer.answer_text) for i, answer in enumerate(answers)])
    rubrics_str = "\n\n".join(["R{}. {} (polarity: {}, meaning: {})\nR{} Examples:\n{}".format(rubric["id"], rubric["title"], rubric["polarity"], rubric['description'], rubric["id"], convert_reasoning_dict(rubric["reasoning_dict"])) for i, rubric in enumerate(rubrics) if rubric['id'] != 0])
    system_prompt = f"""You are an expert instructor for your given course. You've given the short-answer, open-ended question "{question.question_text}" on a recent final exam. You and your expert instructor partner created the following rubrics for this question (labelled R<rubric number> below, along with examples that your partner annotated with reasoning): \n\n{rubrics_str}"""
    
    user_prompt = """
    Given the rubrics mentioned and the following student answers (formatted: <answer ID>. <answer>):\n""" + answers_str + """

    Label and highlight each student's answer based on the rubric(s) that applies to it. Each answer can have multiple rubrics applied, so treat this as a multi-label classification task. Please output in the following format:

    <answer_id>: <comma-separated list of rubrics (labelled R<number>) that apply to this answer> (example: 1: R1, R2, R3)
    """

    # Create a string of existing tags in the format <answer text> [R<rubric number> | <relevancy score> | <reasoning>]
    existing_tags_str = ""
    if (existing_tags is not None):
        # get only tags with relevancy > 0.0
        filtered_tags = [tag for tag in existing_tags.order_by('?') if float(tag.get_reasoning_dict()['relevancy']) > 0.0]
        for tag in filtered_tags[:10]:
            curr_ans = Answer.objects.get(id=tag.answer_id)
            curr_reasoning_dict = tag.get_reasoning_dict()
            if(float(curr_reasoning_dict['relevancy']) > 0.0): existing_tags_str += f"- {curr_ans.answer_text} [Rubric: {tag.tag} | Relevance: {curr_reasoning_dict['relevancy']} | Reason: {curr_reasoning_dict['reasoning']}]\n"

        user_prompt += "\n\n" + "Examples:\n" + existing_tags_str

    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    matching_response = prompt_gpt4(msgs)

    formatting_prompt = """Based on the list of rubrics applied to answers you created, create a JSON dictionary that elaborates on the labels through reasoning and highlighting. Only highlight the most relevant words per rubric that you choose to apply - keep it short! Please provide reasoning for your labels and a relevancy score as well. For the output, create python dictionary that STRICTLY follow the JSON format:

    {"answer_id": [
        {
            "rubric": "<rubric that applies to this answer (labelled R<number>)>",
            "reasoning": "<reason why rubric R<number> applies>",
            "highlighted": "<substring within the answer of 3-6 words that best supports the reasoning>",
            "relevancy": "<0.5 or 1 to indicate partial or full relevance to the answer>"
        },
        {
            "rubric": "<rubric that applies to this answer (labelled R<number>)>",
            "reasoning": "<reason why rubric R<number> applies>",
            "highlighted": "<substring within the answer of 3-6 words that best supports the reasoning>",
            "relevancy": "<0.5 or 1 to indicate partial or full relevance to the answer>"
        },
        ...
    ], ...}"""
    # Selected Answers:\n""" + selection_str

    msgs.append({"role": "assistant", "content": matching_response})
    msgs.append({"role": "user", "content": formatting_prompt})
    response = prompt_gpt4(msgs)

    try:
        tags = json.loads(response)
    except Exception as e:
        print(e)
        tags = []

    return tags

def apply_rubrics_old(question, answers, rubrics):

    answers_str = "\n".join(["{}. {}".format(answer.id, answer.answer_text) for i, answer in enumerate(answers)])
    rubrics_str = "\n".join(["R{}. {} (polarity: {}, meaning: {})".format(rubric["id"], rubric["title"], rubric["polarity"], rubric['description']) for i, rubric in enumerate(rubrics) if rubric['id'] != 0])
    system_prompt = f"""You are an expert instructor for your given course. You've given the short-answer, open-ended question "{question.question_text}" on a recent final exam. You and your expert instructor partner created the following rubrics for this question (labelled R<rubric number> below): \n\n{rubrics_str}"""
    # user_prompt = """Your task is to assign the rubric labels to each of the following students' answers (formatted: <answer ID>. <answer>). Please assign labels to each of the answers provided. Each answer can have multiple rubrics applied too. Treat this as a multi-class classification task. Please provide reasoning for your labels as well. For the output, create a comma-separated list of python dictionaries that STRICTLY follow the JSON format: [{"answer_id": "<id of the answer>", "rubrics": "<comma-separated list of rubrics (labelled R<number>) that apply to this answer>", "reasoning": "<reason you think the rubrics you chose apply to this answer>"}, ...]\n\nStudents' Answers:\n""" + answers_str

    user_prompt = """
    Given the rubrics mentioned and the following student answers (formatted: <answer ID>. <answer>):\n""" + answers_str + """

    Label and highlight each student's answer based on the rubric(s) that applies to it. Each answer can have multiple rubrics applied, so treat this as a multi-label classification task. Only highlight the most relevant words per rubric that you choose to apply - keep it short! Please provide reasoning for your labels as well. For the output, create python dictionary that STRICTLY follow the JSON format:

    {"answer_id": [
        {
            "rubric": "<rubric that applies to this answer (labelled R<number>)>",
            "reasoning": "<reason why rubric R<number> applies>",
            "highlighted": "<substring within the answer of 3-6 words that best supports the reasoning>"
        },
        {
            "rubric": "<rubric that applies to this answer (labelled R<number>)>",
            "reasoning": "<reason why rubric R<number> applies>",
            "highlighted": "<substring within the answer of 3-6 words that best supports the reasoning>"
        },
        ...
    ], ...}
    """

    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    response = prompt_chatgpt(msgs)

    try:
        tags = json.loads(response)
    except Exception as e:
        print(e)
        tags = []

    return tags

def apply_feedback(question, answers, rubrics):

    # convert dict of answer ID keys and reasoning values to a new-line separated string of Answer texts (from DB) and the reasoning
    def convert_reasoning_dict(reasoning_dict):
        reasoning_str = ""
        for answer_id, reasoning in reasoning_dict.items():
            reasoning_str += "{} {}\n".format(Answer.objects.get(id=answer_id).answer_text, f"(REASON: {reasoning})" if reasoning != "" else "")
        return reasoning_str if reasoning_str != "" else "None"

    rubrics_str = "\n\n".join(["R{}. {} (polarity: {}, meaning: {})\nR{} Examples:\n{}".format(rubric["id"], rubric["title"], rubric["polarity"], rubric['description'], rubric["id"], convert_reasoning_dict(rubric["reasoning_dict"])) for i, rubric in enumerate(rubrics) if rubric['id'] != 0])
    system_prompt = f"""You are an expert instructor for your given course. You've given the short-answer, open-ended question "{question.question_text}" on a recent final exam. You and your expert instructor partner created the following rubrics for this question (labelled R<rubric number> below, along with examples that your partner annotated with reasoning): \n\n{rubrics_str}"""

    # get all the associated answer tags per answer --> convert reasoning dicts of each answer tag
    def convert_answer_tags(answer_tags):
        tagged_strs = []
        for tag in answer_tags:
            # e.g. {"rubric": "R1", "reasoning": "", "highlighted": "", "relevancy": "0"}
            reasoning_dict = tag.get_reasoning_dict()
            if (reasoning_dict["relevancy"] != "0"): 
                tagged_strs.append(f"{reasoning_dict['rubric']} (relevance: {reasoning_dict['relevancy']}, reason: {reasoning_dict['reasoning']})")
        
        return "\n".join(tagged_strs)

    full_tags_str = []
    for ans in answers:
        answer_tags = ans.answertag_set.all()
        tags_str = convert_answer_tags(answer_tags)
        curr_tags_str = f"{ans.id}. {ans.answer_text}\n{tags_str}"
        full_tags_str.append(curr_tags_str)
    
    answers_str = "\n\n".join(full_tags_str)

    user_prompt = """Based on the rubrics mentioned, you now have the following student answers (formatted: <answer ID>. <answer>, along with annotated rubrics you recently associated with each underneath it):\n\n""" + answers_str + """

    Provide feedback and list the connected associations for each student's answer based on the rubric(s) that applies to it. Each piece of feedback can have multiple associations between it and the answer itself - these will be used to highlight parts to the students. Only associate/highlight the most relevant words - keep it short! For the output, create python dictionary that STRICTLY follow the JSON format:

    {"answer_id": {
        "feedback": "<constructive and helpful feedback that you'd give the student based on the rubrics attached to the answer - try to understand the internal needs of the student instead of just saying what is missing from the answer>",
        "associations": [<list of dicts in the format: {"answer_highlight": "<4-7 words from the answer>", "feedback_highlight": "<4-7 words from the feedback>"}. Can be one or more. This is meant to show association between which part of the answer was commented upon>],
    }, ...}
    """

    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    response = prompt_gpt4(msgs)

    try:
        feedbacks = json.loads(response)
    except Exception as e:
        print(e)
        feedbacks = []

    return feedbacks

def apply_feedback_full(question, answers, rubrics):

    # convert dict of answer ID keys and reasoning values to a new-line separated string of Answer texts (from DB) and the reasoning
    def convert_reasoning_dict(reasoning_dict):
        reasoning_str = ""
        for answer_id, reasoning in reasoning_dict.items():
            reasoning_str += "{} {}\n".format(Answer.objects.get(id=answer_id).answer_text, f"(REASON: {reasoning})" if reasoning != "" else "")
        return reasoning_str if reasoning_str != "" else "None"

    rubrics_str = "\n\n".join(["R{}. {} (polarity: {}, meaning: {})\nR{} Examples:\n{}".format(rubric["id"], rubric["title"], rubric["polarity"], rubric['description'], rubric["id"], convert_reasoning_dict(rubric["reasoning_dict"])) for i, rubric in enumerate(rubrics) if rubric['id'] != 0])
    system_prompt = f"""You are an expert instructor for your given course. You've given the short-answer, open-ended question "{question.question_text}" on a recent final exam. You and your expert instructor partner created the following rubrics for this question (labelled R<rubric number> below, along with examples that your partner annotated with reasoning): \n\n{rubrics_str}"""

    # get all the associated answer tags per answer --> convert reasoning dicts of each answer tag
    def convert_answer_tags(answer_tags):
        tagged_strs = []
        for tag in answer_tags:
            # e.g. {"rubric": "R1", "reasoning": "", "highlighted": "", "relevancy": "0"}
            reasoning_dict = tag.get_reasoning_dict()
            if (reasoning_dict["relevancy"] != "0"): 
                tagged_strs.append(f"{reasoning_dict['rubric']} (relevance: {reasoning_dict['relevancy']}, reason: {reasoning_dict['reasoning']})")
        
        return "\n".join(tagged_strs)

    full_tags_str = []
    for ans in answers:
        answer_tags = ans.answertag_set.all()
        tags_str = convert_answer_tags(answer_tags)
        curr_tags_str = f"{ans.id}. {ans.answer_text}\n{tags_str}"
        full_tags_str.append(curr_tags_str)
    
    answers_str = "\n\n".join(full_tags_str)

    user_prompt = """Based on the rubrics mentioned, you now have the following student answers (formatted: <answer ID>. <answer>, along with annotated rubrics you recently associated with each underneath it):\n\n""" + answers_str + """

    Provide feedback and list the connected associations for each student's answer based on the rubric(s) that applies to it. Each piece of feedback can have multiple associations between it and the answer itself - these will be used to highlight parts to the students. Only associate/highlight the most relevant words - keep it short! For the output, create python dictionary that STRICTLY follow the JSON format:

    {"answer_id": {
        "feedback": "<constructive and helpful feedback that you'd give the student based on the rubrics attached to the answer - try to understand the internal needs of the student instead of just saying what is missing from the answer>",
        "associations": [<list of dicts in the format: {"answer_highlight": "<4-7 words from the answer>", "feedback_highlight": "<4-7 words from the feedback>"}. Can be one or more. This is meant to show association between which part of the answer was commented upon>],
    }, ...}
    """

    # Create a string of existing feedback in the format <answer text> [Feedback: <feedback> | AnsH: <answer_highlight> | FeedH: <feedback_highlight>]
    existing_feedback_str = ""
    for ans in answers:
        answer_feedbacks = ans.answerfeedback_set.all()
        for feedback in answer_feedbacks[:1]:
            existing_feedback_str += f"- {ans.answer_text} [Feedback: {feedback.feedback} | AnsH: {feedback.answer_highlight} | FeedH: {feedback.feedback_highlight}]\n"
    
    if (existing_feedback_str): user_prompt += "\n\n" + "Example Feedbacks:\n" + existing_feedback_str

    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    response = prompt_gpt4(msgs)

    try:
        feedbacks = json.loads(response)
    except Exception as e:
        print(e)
        feedbacks = []

    return feedbacks

def main():
    response = prompt_gpt4([{"role": "system", "content": "You are a wise professor in HCI"}, {"role": "user", "content": "Give your students your top 3 tips for success in this class."}])

    print(response)

if __name__ == "__main__":
    main()