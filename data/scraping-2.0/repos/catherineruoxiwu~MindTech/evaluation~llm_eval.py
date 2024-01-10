import json
import os
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI

client = OpenAI()

directory_path = "./out"
exclude = ["gpt3.5-turbo.json"]

num_questions = 99

rubric_criteria = ["Establishing_Rapport", "Active_Listening", "Verbal_Communication", "Assessment_and_Goal_Setting", "Interventions_and_Techniques", "Boundaries_and_Ethics", "Cultural_Competence", "User_Experience"]
eval_rubric = """**Therapist Bot Conversation Evaluation Rubric**
    1. Establishing_Rapport (1-5) 
        - Demonstrates an ability to establish trust and rapport with the user.
        - Creates a safe and non-judgmental virtual environment. 
        - Uses welcoming and empathetic language. 
    2. Active_Listening (1-5) 
        - Demonstrates active listening skills through appropriate responses. 
        - Responds with empathy and validation. 
        - Asks clarifying questions and encourages the user to share more. 
    3. Verbal_Communication (1-5) 
        - Uses clear and concise language. 
        - Manages pacing and tone appropriately. 
    4. Assessment_and_Goal_Setting (1-5) 
        - Gathers user information effectively to understand their needs and goals. 
        - Collaboratively sets clear and achievable goals for the conversation. 
        - Provides information about the therapy process. 
    5. Interventions_and_Techniques (1-5) 
        - Applies appropriate therapeutic interventions based on the user's input. 
        - Adapts responses to the user's changing emotional states and needs. 
        - Provides useful coping strategies or resources when appropriate. 
    6. Boundaries_and_Ethics (1-5) 
        - Maintains appropriate boundaries and confidentiality. 
        - Demonstrates an understanding of and adherence to ethical guidelines for AI therapy. 
        - Addresses potential ethical issues, such as user privacy and data security. 
    7. Cultural_Competence (1-5) 
        - Shows cultural sensitivity and awareness in responses. 
        - Adapts responses to consider the user's cultural background or preferences. 
        - Addresses diversity-related issues respectfully. 
    8. User_Experience (1-5) 
        - Provides a user-friendly and engaging experience. 
        - Promotes a sense of comfort and ease during the conversation. 
        - Encourages the user to return for further interactions.
        - Monitors user feedback and adjusts responses accordingly.
        - Measures user satisfaction and perceived benefits. 
        - Demonstrates improvements in user well-being or mental health. 
    10. Overall Score (1-5)
        - Average the scores from each category to determine the overall rating for the therapist bot's conversation.
    """

def evaluate_QA(qa_pairs):
    """
    Args:
        qa_pairs: list of objects {"model_name": ..., "question": ..., "answer": ...}
    """
    num_models = len(qa_pairs)
    role_prompt = f"""Please act as an impartial judge and evaluate the quality of the responses provided by {num_models} AI psychiatrist models to the client's complaint or questions. You will only be given the first few sentences of the conversation.
    Your evaluation should be based solely on the consultation rubric provided at the end, titled "Therapist Bot Conversation Evaluation Rubric". You cannot solely judge the quality based on "whether or not more advice or suggestions are given". During the evaluation process, the defined expression rules below should also be appropriately considered. For each of the {num_models} question-answer pairs, produce separate evaluation rubric results. After evaluating the {num_models} models, decide which AI psychiatrist model is the best model.

    Your response should be in JSON format. The output JSON format should be:
        {{"rubric_results": {{INSERT_MODEL_NAME_HERE: INSERT_RUBRIC_RESULT_HERE, ...}},
          "best_model_idx": INSERT_INDEX_OF_BEST_MODEL,
          "best_model_name": INSERT_INDEX_OF_BEST_MODEL }}
    The keys in the rubric result should be {str(rubric_criteria)}.
     
    {eval_rubric}
    """

    eval_prompt = ""
    for i, qa_pair in enumerate(qa_pairs):
        model_name, question, answer = qa_pair["model_name"], qa_pair["question"], qa_pair["answer"]
        eval_prompt += f"AI psychiatrist model #{i}:\n\tModel name: {model_name}\n\tQuestion: {question}\n\tAnswer: {answer}\n\n"

    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": role_prompt},
            {"role": "user", "content": eval_prompt}
        ],
        response_format={"type": "json_object"}
    )

    return json.loads(completion.choices[0].message.content)

def evaluate_models(num_questions, model_results):
    """
    Args:
        model_results: list of {"model_name": ...,
                                "1": {"question": ..., "answer": ...}, 
                                "2": {"question": ..., "answer": ...}, ...}
    """
    eval_stat = {}
    model_names = [model["model_name"] for model in model_results]

    for name in model_names:
        eval_stat[name] = {
            "num_best": 0,
        }
        for criterion in rubric_criteria:
            eval_stat[name][criterion] = 0

    for i in range(1, num_questions + 1):
        qa_pairs = []
        for model in model_results:
            pair = model[str(i)]
            pair["model_name"] = model["model_name"]
            qa_pairs.append(pair)
        
        eval_res = evaluate_QA(qa_pairs)
        print(json.dumps(eval_res, indent=4))

        best_model = eval_res["best_model_name"]
        if best_model in model_names:
            eval_stat[eval_res["best_model_name"]]["num_best"] += 1
        for model_name, res in eval_res["rubric_results"].items():
            for criterion in rubric_criteria:
                eval_stat[model_name][criterion] += res[criterion]

    # Compute the average evaluation scores
    for name in model_names:
        for criterion in rubric_criteria:
            eval_stat[name][criterion] /= num_questions

    print(json.dumps(eval_stat, indent=4))
    return eval_stat

def plot_evaluation_chart(eval_stat):
    """
    Takes a dictionary of evaluation statistics and plots a bar chart.

    Args:
    eval_stat (dict): A dictionary containing evaluation statistics for various models.

    Returns:
    None: This function plots a bar chart.
    """
    # Extracting categories and scores for each model
    categories = list(eval_stat[next(iter(eval_stat))].keys())[1:]  # Excluding 'num_best'
    models = list(eval_stat.keys())
    scores = {model: [eval_stat[model][cat] for cat in categories] for model in models}

    # Number of categories
    n_categories = len(categories)

    # X locations for the groups
    ind = np.arange(n_categories)  
    width = 0.25  # the width of the bars

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))

    # Bar charts
    for i, model in enumerate(models):
        ax.bar(ind + i*width, scores[model], width, label=model)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Comparison of Scores by Category and Model')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.legend()

    # Show the plot
    plt.tight_layout()
    plt.savefig('eval_result.png')


def parse_input_files(exclude):
    model_results = []
    idx_name = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".json") and filename not in exclude:
            model_name = os.path.splitext(os.path.basename(filename))[0]
            file_path = os.path.join(directory_path, filename)
            with open(file_path, "r") as file:
                model_result = json.load(file)
                model_result["model_name"] = model_name
                idx_name.append(model_name)
                model_results.append(model_result)
    return idx_name, model_results


if __name__ == "__main__":
    idx_name, model_results = parse_input_files(exclude)
    eval_stat = evaluate_models(num_questions, model_results)
    plot_evaluation_chart(eval_stat)
