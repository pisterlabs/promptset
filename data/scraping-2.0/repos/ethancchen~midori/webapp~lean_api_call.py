from openai import OpenAI
import json

# openai.api_key = "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"

def get_completion(prompt, engine = 'text-davinci-003'):
    client = OpenAI()
    # OpenAI.api_key = "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"
    print(OpenAI.api_key)
    response = client.completions.create(
        model = engine,
        prompt = prompt,
        max_tokens = 3000,
        n = 1
    )
    return response.choices[0].text

def generate_ans(text):
    prompt=f"""

    ```{text}```

    Generate a concise and informative lean canvas based on the following problem statement and potential solution. Provide responses within the specified character limits and formats.
    make sure that you provide answer to each of these prompts on a new line

    Specific prompts:

    Problem summary: Summarize the problem in under 450 characters.
    Solution summary: Summarize the solution in under 200 characters.
    Unique value proposition: What's the unique value proposition for the provided solution? Answer in 1 sentence under 450 characters.
    Key metrics: What are the key metrics for the provided solution? Answer in paragraph form under 200 characters.
    Unfair advantages: What are some unfair advantages that other companies may have that could affect the effectiveness of the solution? Answer in paragraph form under 200 characters.
    Channels: How can one provide this solution to customers? Answer in paragraph form under 200 characters.
    Customer segments: Who are the target customers? Answer in one sentence under 450 characters.
    Cost structure: What is the cost structure for this solution? Answer in one sentence under 470 characters.
    Revenue streams: What are different types of revenue streams for this solution? Answer in paragraph form under 470 characters.

    """

    # print(prompt)
    ans = get_completion(prompt)

    return ans


def get_data(text):

    ans= generate_ans(text)
    lines = [line for line in ans.splitlines() if line.strip()]

    problem_summary = lines[0]
    # print(problem_summary)
    solution_summary = lines[1]
    # print(solution_summary)
    uniq_val_prop = lines[2]
    key_metrics = lines[3]
    unfair_advtg = lines[4]
    channels = lines[5]
    customer_seg = lines[6]
    cost_struct = lines[7]
    revenue_streams = lines[8]


    return problem_summary,solution_summary, uniq_val_prop, key_metrics, unfair_advtg, channels, customer_seg,cost_struct, revenue_streams
