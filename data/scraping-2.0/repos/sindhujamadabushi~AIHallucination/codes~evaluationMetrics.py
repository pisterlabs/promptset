from openai import OpenAI

def exact_match(ground_truth, answer):
    if ground_truth in answer:
        return 'no'                 # case model not hallucinating
    else:
        return 'yes'                # case model is hallucinating

def llm_selfevaluation(context, ground_truth, answer, openai_key):
    client = OpenAI(api_key = openai_key)
    response = []
    response = client.completions.create(
        model="text-davinci-002",
        prompt=f"Determine the consistency of the answer with the ground truth with a 'yes' or 'no' response. Note that consistency measures how much information in the ground truth is present in the answer. The answer can be in different formats. \nContext: {context} Ground Truth: {ground_truth}\nAnswer: {answer}yes/no:",
        max_tokens = 10
    )
    score = response.choices[0].text.strip().split("\\n")
    return score[0]



